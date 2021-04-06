import torch
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import pandas as pd
from pycox.models.loss import rank_loss_deephit_single


def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def make_norm(state):
    state_size = state.numel()
    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm

     
        
class Encoder(nn.Module):
    """
        Encoding of the initial values of the memory states
        Input: 
            - number of covariates
            - number of memory states
            - hidden layer neurons, given as array (e.g. [10,10,10])
            - dropout for hidden layers, given as array (e.g. [0.2,0.3,0.4])
    """
    def __init__(self,num_in,num_latent,layers,p_dropout):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(*((nn.Linear(num_in,layers[0]), nn.ReLU(), nn.Dropout(p_dropout[0])) + tuple(tup for element in tuple(((nn.Linear(layers[i],layers[i+1]), nn.ReLU(), nn.Dropout(p_dropout[i+1])) for i in range(len(layers)-1))) for tup in element) + (nn.Linear(layers[-1],num_latent),)))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    def forward(self, x):
        return self.net(x)


class ODEFunc(nn.Module):
    """
        KFE_KBE function to calculate the derivatives in the ODEsolver
        Input: 
            - transition matrix giving possible transitions (e.g. [[NA,1,1],[NA,NA,1],[NA,NA,NA]] for the illness-death model)
            - number of covariates
            - number of memory states
            - hidden layer neurons, given as array (e.g. [10,10,10])
            - dropout for hidden layers, given as array (e.g. [0.2,0.3,0.4])
            - softplus parameter (should be left at 1.)
    """
    def __init__(self,transition_matrix,num_in,num_latent,layers,softplus_beta=1.):
        super(ODEFunc, self).__init__()
        
        self.softplus_beta = softplus_beta
        self.transition_matrix = transition_matrix
        self.trans_dim = transition_matrix.shape[0]
        self.num_latent = num_latent
        self.number_of_hazards = int(np.nansum(transition_matrix.flatten().cpu()))
        self.num_probs = np.prod(transition_matrix.shape)
        # use this NN if covariates are to be included
        self.net = nn.Sequential(*((nn.Linear(2*self.num_probs+self.number_of_hazards+2*num_latent+num_in+1,layers[0]), nn.Tanh()) + tuple(tup for element in tuple(((nn.Linear(layers[i],layers[i+1]), nn.Tanh()) for i in range(len(layers)-1))) for tup in element) + (nn.Linear(layers[-1],self.number_of_hazards+num_latent),)))
        # use this NN if memory is included
        # self.net = nn.Sequential(*((nn.Linear(2*self.num_probs+self.number_of_hazards+2*num_latent+1,layers[0]), nn.Tanh()) + tuple(tup for element in tuple(((nn.Linear(layers[i],layers[i+1]), nn.Tanh()) for i in range(len(layers)-1))) for tup in element) + (nn.Linear(layers[-1],self.number_of_hazards+num_latent),)))

        count = 0
        length = len(list(self.net.modules()))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0.)
            if count==length-1:
                nn.init.constant_(m.weight, 0)
            count += 1
        self.num_in = num_in
    
    def set_x(self, x):
        self.x = x

    def set_y0(self, y0):
        self.y0 = y0

    # KFE_KBE function
    def forward(self, t, y):
        # pass values through NN
        # out = self.net(torch.cat((y,self.y0,torch.tensor([t],device=y.device).repeat((y.shape[0],1))),1))
        out = self.net(torch.cat((y,self.y0,self.x,torch.tensor([t],device=y.device).repeat((y.shape[0],1))),1))
        
        # build Q matrix from output
        qvec = torch.nn.functional.softplus(out[:,:self.number_of_hazards],beta=self.softplus_beta)
        q = torch.zeros(self.trans_dim, self.trans_dim,device=y.device).repeat((y.shape[0],1,1))
        q[self.transition_matrix.repeat((y.shape[0],1,1))==1] = qvec.flatten()
        q[torch.eye(self.trans_dim, self.trans_dim,device=y.device).repeat((y.shape[0],1,1)) == 1] = -torch.sum(q,2).flatten()
        
        # get P matrix
        P = torch.reshape(y[:,:self.num_probs],(y.shape[0],self.trans_dim,self.trans_dim))
        P_back = torch.reshape(y[:,self.num_probs:(2*self.num_probs)],(y.shape[0],self.trans_dim,self.trans_dim))
        # calculate right side of KFE and KBE
        Pprime = torch.bmm(P, q)
        Pprime_back = -torch.bmm(q, P_back)
        return torch.cat((Pprime.reshape(y.shape[0],self.num_probs),Pprime_back.reshape(y.shape[0],self.num_probs),qvec,out[:,self.number_of_hazards:]),1)
     
class ODEBlock(nn.Module):
    """
        Helper Function to define the initial value problem
    """
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.num_probs = odefunc.num_probs
        self.trans_dim = odefunc.trans_dim
        self.number_of_hazards = odefunc.number_of_hazards
        self.transition_matrix = odefunc.transition_matrix

    def forward(self, y0, x, tinterval):
        self.odefunc.set_x(x) # saves covariates
        self.odefunc.set_y0(y0)
        p0 = torch.eye(self.trans_dim,device=y0.device).reshape(self.num_probs).repeat((y0.shape[0],1))
        Q0 = torch.zeros(self.number_of_hazards,device=x.device).repeat((y0.shape[0],1))
        yin = torch.cat((p0,p0,Q0,y0),1)
        out = odeint(self.odefunc, yin, tinterval, method="dopri5", atol=1e-8, rtol=1e-8) #,adjoint_options=dict(norm=make_norm(y0)))
        return out       
        
class SurvNODE(nn.Module):
    """
        SurvNODE class, 
    """
    def __init__(self,odeblock,encoder):
        super(SurvNODE, self).__init__()
        self.odeblock = odeblock
        self.encoder = encoder
        self.num_probs = odeblock.num_probs
        self.trans_dim = odeblock.trans_dim
        self.number_of_hazards = odeblock.number_of_hazards
        self.transition_matrix = odeblock.transition_matrix

    def forward(self, x, tstart, tstop, from_state, to_state):
        # get P_ij(0,t) and P_ij(t,0) at all batch times
        out = self.odeblock(self.encoder(x),x,torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])))
        
        # get P_ij(s,t) through Kolmogorov backward equation
        tstart_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero(as_tuple=False) for time in tstart]))
        Ttstartinv = torch.cat([out[tstart_indices[i],i,self.num_probs:(self.num_probs*2)] for i in range(len(tstart))]).reshape((tstart.shape[0],self.trans_dim,self.trans_dim))
        tstop_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero(as_tuple=False) for time in tstop]))
        Ttstop = torch.cat([out[tstop_indices[i],i,:self.num_probs] for i in range(len(tstop))]).reshape((tstop.shape[0],self.trans_dim,self.trans_dim))
        S = torch.bmm(Ttstartinv,Ttstop)
        S = torch.cat([S[i:i+1,from_state[i]-1,from_state[i]-1] for i in range(len(from_state))])
        
        
#         # get P_ij(s,0) by inverting P_ij(0,s)
#         tstart_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero() for time in tstart]))
#         Ttstart = out[tstart_indices,[i for i in range(tstart.shape[0])],:self.num_probs].reshape((tstart.shape[0],self.trans_dim,self.trans_dim))
#         Ttstartinv = torch.inverse(Ttstart)
#         # # inverse with conditioning (?)
#         # Ttstartinv = torch.inverse(Ttstart+1e-5*torch.eye(Ttstart.shape[1],device=x.device).flatten().repeat(Ttstart.shape[0]).reshape(Ttstart.shape))
#         tstop_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero() for time in tstop]))
#         Ttstop = out[tstop_indices,[i for i in range(tstop.shape[0])],:self.num_probs].reshape((tstop.shape[0],self.trans_dim,self.trans_dim))
#         S = torch.bmm(Ttstartinv,Ttstop)
#         S = torch.cat([S[i:i+1,from_state[i]-1,from_state[i]-1] for i in range(len(from_state))])
        
        
        # get lambda at tstop
        net_in = torch.cat((torch.cat([out[tstop_indices[i],i:i+1,:] for i in range(len(tstop))]),self.encoder(x),x,tstop.reshape(-1,1)),1)
        # net_in = torch.cat((torch.cat([out[tstop_indices[i],i:i+1,:] for i in range(len(tstop))]),self.encoder(x),tstop.reshape(-1,1)),1)
        qvec = torch.nn.functional.softplus(self.odeblock.odefunc.net(net_in)[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta)
        q = torch.zeros(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1))
        q[self.transition_matrix.repeat((x.shape[0],1,1))==1] = qvec.flatten()
        lam = torch.cat([q[t:t+1,from_state[t]-1,to_state[t]-1] for t in range(len(from_state))])
        # get all augmented hazards at the final time (t=multiplier) for loss term
        # run this with covariatios
        net_in = torch.cat((out[-1,:,:],self.encoder(x),x, torch.tensor([max(tstop)],device=x.device).repeat(tstop.reshape(-1,1).shape)),1)
        
        # net_in = torch.cat((out[-1,:,:],self.encoder(x),torch.tensor([max(tstop)],device=x.device).repeat(tstop.reshape(-1,1).shape)),1)
        
        out = self.odeblock.odefunc.net(net_in)
#         all_hazards_T = torch.cat((torch.nn.functional.softplus(out[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta),out[:,self.number_of_hazards:]),-1)
        all_hazards_T = out[:,self.number_of_hazards:]
    
        return (S,lam,all_hazards_T)
    
    def predict(self,x,tvec):
        """
            Prediction of survival based on covariates x at times in tvec.
            This function returns the transition matrix P_ij(0,t) at every t in tvec.
        """
        with torch.no_grad():
            out = self.odeblock(self.encoder(x),x,tvec.float().to(x.device))
            T = out[:,:,:self.odeblock.odefunc.num_probs].reshape((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim))
        return T
    
    def predict_hazard(self,x,tvec):
        """
            Predict cause specific hazard function based on covariates x at times in tvec.
            This function returns the matrix Q of instantaneous hazards over time.
        """
        with torch.no_grad():
            tvec = tvec.float().to(x.device)
            out = self.odeblock(self.encoder(x),x,tvec)
            Qvec = torch.zeros((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim),device=x.device)
            for i in range(tvec.shape[0]):
                net_in = torch.cat((out[i,:,:], self.encoder(x), x ,tvec[i].repeat(x.shape[0]).reshape(-1, 1)),1)
                temp = self.odeblock.odefunc.net(net_in)
                qvec = torch.nn.functional.softplus(temp[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta)
                Q = torch.zeros(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1))
                Q[self.transition_matrix.repeat((x.shape[0],1,1))==1] = qvec.flatten()
                Q[torch.eye(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1)) == 1] = -torch.sum(Q,2).flatten()
                Qvec[i,:,:,:] = Q
        return Qvec

    # def predict_hazard(self,x,tvec):
    #     """
    #         Predict cause specific hazard function based on covariates x at times in tvec.
    #         This function returns the matrix Q of instantaneous hazards over time.
    #     """
    #     with torch.no_grad():
    #         tvec = tvec.float().to(x.device)
    #         out = self.odeblock(self.encoder(x),x,tvec)
    #         Qvec = torch.zeros((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim),device=x.device)
    #         for i in range(tvec.shape[0]):
    #             net_in = torch.cat((out[i,:,:-1],self.encoder(x),tvec[i].repeat(x.shape[0]).reshape(-1,1)),1)
    #             temp = self.odeblock.odefunc.net(net_in)
    #             qvec = torch.nn.functional.softplus(temp[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta)
    #             Q = torch.zeros(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1))
    #             Q[self.transition_matrix.repeat((x.shape[0],1,1))==1] = qvec.flatten()
    #             Q[torch.eye(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1)) == 1] = -torch.sum(Q,2).flatten()
    #             Qvec[i,:,:,:] = Q
    #     return Qvec
    
    def predict_cumhazard(self,x,tvec):
        """
            Predict cumulative hazard function based on covariates x at times in tvec.
            The cumulative cause specific hazards are given as the integral from 0 to t over the cause specific hazards.
            This function returns a vector of cause specific cumulative hazards over time.
        """
        with torch.no_grad():
            tvec = tvec.float().to(x.device)
            tvec = torch.unique(torch.cat([torch.tensor([0.],device=x.device),tvec]))
            out = self.odeblock(self.encoder(x),x,tvec)
            qvec = out[:,:,(2*self.num_probs):(2*self.num_probs+self.number_of_hazards)]
            Qvec = torch.zeros((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim),device=x.device)
            Qvec[self.transition_matrix.repeat((tvec.shape[0],x.shape[0],1,1))==1] = qvec.flatten()
        return Qvec

            
def loss(odesurv,x,Tstart,Tstop,From,To,trans,status,mu=1e-4):
    """
        Loss function
        Parameter mu regulates the influence of the Lyapunov loss
    """
    trans_exist = torch.tensor([odesurv.transition_matrix[From[i]-1,To[i]-1] for i in range(len(From))])
    trans_exist = torch.where(trans_exist==1)
    x = x[trans_exist]
    Tstart = Tstart[trans_exist]
    Tstop = Tstop[trans_exist]
    From = From[trans_exist]
    To = To[trans_exist]
    status = status[trans_exist]
    
    S,lam,all_h_T = odesurv(x,Tstart,Tstop,From,To)
    loglik = -(status*torch.log(lam)+torch.log(S)).mean()
    reg = torch.norm(all_h_T,2,dim=1).mean()
    # ranking_loss = rank_loss_deephit_single()

    return (loglik + mu*reg), loglik, reg