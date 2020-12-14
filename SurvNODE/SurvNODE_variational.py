import torch
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import seaborn as sns
import numpy as np
import pandas as pd


class Prior(nn.Module):
    """ 
        Encoding of mean and standard deviation for the prior
    """
    def __init__(self,num_in,num_latent,layers,p_dropout):
        super(Prior, self).__init__()
        self.net = nn.Sequential(*((nn.Linear(num_in,layers[0]), nn.ReLU(), nn.Dropout(p_dropout[0])) + tuple(tup for element in tuple(((nn.Linear(layers[i],layers[i+1]), nn.ReLU(), nn.Dropout(p_dropout[i+1])) for i in range(len(layers)-1))) for tup in element) + (nn.Linear(layers[-1],num_latent),)))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, x):
        return self.net(x)

class Post(nn.Module):
    """ 
        Encoding of mean and standard deviation for the variational approximation of the posterior
    """
    def __init__(self,num_in,num_latent,layers,p_dropout):
        super(Post, self).__init__()
        self.num_in = num_in
        self.net = nn.Sequential(*((nn.Linear(num_in+2,layers[0]), nn.ReLU(), nn.Dropout(p_dropout[0])) + tuple(tup for element in tuple(((nn.Linear(layers[i],layers[i+1]), nn.ReLU(), nn.Dropout(p_dropout[0])) for i in range(len(layers)-1))) for tup in element) + (nn.Linear(layers[-1],num_latent),)))        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        return self.net(x)

class ODEFunc(nn.Module):
    """ 
        KFE_KBE class
    """
    def __init__(self,transition_matrix,num_in,num_latent,layers,p_dropout,softplus_beta=1.):
        super(ODEFunc, self).__init__()
        
        self.softplus_beta = softplus_beta
        self.transition_matrix = transition_matrix
        self.trans_dim = transition_matrix.shape[0]
        self.num_latent = num_latent
        self.number_of_hazards = int(np.nansum(transition_matrix.flatten().cpu()))
        self.num_probs = np.prod(transition_matrix.shape)
        self.net = nn.Sequential(*((nn.Linear(2*self.num_probs+self.number_of_hazards+2*num_latent+1,layers[0]), nn.Tanh()) + tuple(tup for element in tuple(((nn.Linear(layers[i],layers[i+1]), nn.Tanh()) for i in range(len(layers)-1))) for tup in element) + (nn.Linear(layers[-1],self.number_of_hazards+num_latent),)))
        count = 0
        length = len(list(self.net.modules()))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0.)
            if count==length-1:
                nn.init.normal_(m.weight, mean=0, std=0)
            count += 1
        self.num_in = num_in
    
    def set(self, y0):
        self.y0 = y0

    def forward(self, t, y):
        """ 
            KFE_KBE function
        """
        out = self.net(torch.cat((y,self.y0,torch.tensor([t],device=y.device).repeat((y.shape[0],1))),1))
        qvec = torch.nn.functional.softplus(out[:,:self.number_of_hazards],beta=self.softplus_beta)
        q = torch.zeros(self.trans_dim, self.trans_dim,device=y.device).repeat((y.shape[0],1,1))
        q[self.transition_matrix.repeat((y.shape[0],1,1))==1] = qvec.flatten()
        q[torch.eye(self.trans_dim, self.trans_dim,device=y.device).repeat((y.shape[0],1,1)) == 1] = -torch.sum(q,2).flatten()
        P = torch.reshape(y[:,:self.num_probs],(y.shape[0],self.trans_dim,self.trans_dim))
        P_back = torch.reshape(y[:,self.num_probs:(2*self.num_probs)],(y.shape[0],self.trans_dim,self.trans_dim))
        Pprime = torch.bmm(P, q)
        Pprime_back = -torch.bmm(q, P_back)
        return torch.cat((Pprime.reshape(y.shape[0],self.num_probs),Pprime_back.reshape(y.shape[0],self.num_probs),qvec,out[:,self.number_of_hazards:]),1)
        
class ODEBlock(nn.Module):
    """ 
        Helpfer function for initial value problem
    """
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.num_probs = odefunc.num_probs
        self.trans_dim = odefunc.trans_dim
        self.number_of_hazards = odefunc.number_of_hazards
        self.transition_matrix = odefunc.transition_matrix

    def forward(self, y0, tinterval):
        self.odefunc.set(y0)
        p0 = torch.eye(self.trans_dim,device=y0.device).reshape(self.num_probs).repeat((y0.shape[0],1))
        Q0 = torch.zeros(self.number_of_hazards,device=y0.device).repeat((y0.shape[0],1))
        yin = torch.cat((p0,p0,Q0,y0),1)
        out = odeint(self.odefunc, yin, tinterval, method="dopri5", atol=1e-8, rtol=1e-8)
        return out       
    
def normal_kl(mu1, lv1, mu2, lv2):
    """
        Code from latent ODE example https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class VarSurvNODE(nn.Module):
    """
        Variational implementation of the SurvNODE method.
    """
    def __init__(self,odeblock,prior_mean,prior_var,post_mean,post_var):
        super(VarSurvNODE, self).__init__()
        self.odeblock = odeblock
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.post_mean = post_mean
        self.post_var = post_var
        self.num_probs = odeblock.num_probs
        self.trans_dim = odeblock.trans_dim
        self.number_of_hazards = odeblock.number_of_hazards
        self.transition_matrix = odeblock.transition_matrix

    def forward(self, x, tstart, tstop, from_state, to_state, status):        
        # sample from prior
        pz0_mean = self.prior_mean(x)
        pz0_logvar = self.prior_var(x)
        epsilon = torch.randn(pz0_mean.size()).to(x.device)
        z0prior = epsilon * torch.exp(.5 * pz0_logvar) + pz0_mean

        qz0_mean = self.post_mean(torch.cat([tstop.reshape(-1,1),status.reshape(-1,1),x],dim=1))
        qz0_logvar = self.post_var(torch.cat([tstop.reshape(-1,1),status.reshape(-1,1),x],dim=1))
        epsilon = torch.randn(qz0_mean.size()).to(x.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
    
        out = self.odeblock(z0,torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])))
        
        # P_ij(s,0) by Kolmogorov backward equation
        tstart_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero() for time in tstart]))
        Ttstartinv = torch.cat([out[tstart_indices[i],i,self.num_probs:(self.num_probs*2)] for i in range(len(tstart))]).reshape((tstart.shape[0],self.trans_dim,self.trans_dim))
        tstop_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero() for time in tstop]))
        Ttstop = torch.cat([out[tstop_indices[i],i,:self.num_probs] for i in range(len(tstop))]).reshape((tstop.shape[0],self.trans_dim,self.trans_dim))
        S = torch.einsum("ijk,ikl->ijl",(Ttstartinv,Ttstop))
        S = torch.cat([S[i:i+1,from_state[i]-1,from_state[i]-1] for i in range(len(from_state))])

#         # P_ij(s,0) by direct matrix inversion
#         tstart_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero() for time in tstart]))
#         Ttstart = out[tstart_indices,[i for i in range(tstart.shape[0])],:self.num_probs].reshape((tstart.shape[0],self.trans_dim,self.trans_dim))
#         Ttstartinv = torch.inverse(Ttstart)
#         tstop_indices = torch.flatten(torch.cat([(torch.unique(torch.cat([torch.tensor([0.],device=x.device),tstart,tstop])) == time).nonzero() for time in tstop]))
#         Ttstop = out[tstop_indices,[i for i in range(tstop.shape[0])],:self.num_probs].reshape((tstop.shape[0],self.trans_dim,self.trans_dim))
#         S = torch.einsum("ijk,ikl->ijl",(Ttstop,Ttstartinv))
#         S = torch.cat([S[i:i+1,from_state[i]-1,from_state[i]-1] for i in range(len(from_state))])
        
        # get lambda at tstop
        net_in = torch.cat((torch.cat([out[tstop_indices[i],i:i+1,:] for i in range(len(tstop))]),z0,tstop.reshape(-1,1)),1)
        qvec = torch.nn.functional.softplus(self.odeblock.odefunc.net(net_in)[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta)
        q = torch.zeros(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1))
        q[self.transition_matrix.repeat((x.shape[0],1,1))==1] = qvec.flatten()
        lam = torch.cat([q[t:t+1,from_state[t]-1,to_state[t]-1] for t in range(len(from_state))])
        # get all augmented hazards at the final time for loss term
        net_in = torch.cat((out[-1,:,:],z0,torch.tensor([max(tstop)],device=x.device).repeat(tstop.reshape(-1,1).shape)),1)
        out = self.odeblock.odefunc.net(net_in)
        all_hazards_T = torch.cat((torch.nn.functional.softplus(out[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta),out[:,self.number_of_hazards:]),-1)
        
        return (analytic_kl,S,lam,all_hazards_T)
    
    def predict(self,x,tvec):
        with torch.no_grad():
            pz0_mean = self.prior_mean(x)
            pz0_logvar = self.prior_var(x)
            epsilon = torch.randn(pz0_mean.size()).to(x.device)
            z0_prior = epsilon * torch.exp(.5 * pz0_logvar) + pz0_mean
            out = self.odeblock(z0_prior,tvec.float().to(x.device))
            T = out[:,:,:self.odeblock.odefunc.num_probs].reshape((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim))
        return T
    
    def predict_hazard(self,x,tvec):
        with torch.no_grad():
            tvec = tvec.float().to(x.device)
            
            pz0_mean = self.prior_mean(x)
            pz0_logvar = self.prior_var(x)
            epsilon = torch.randn(pz0_mean.size()).to(x.device)
            z0prior = epsilon * torch.exp(.5 * pz0_logvar) + pz0_mean
          
            out = self.odeblock(z0prior,tvec)
            Qvec = torch.zeros((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim),device=x.device)
            for i in range(tvec.shape[0]):
                net_in = torch.cat((out[i,:,:],z0,tvec[i].repeat(x.shape)),1)
                temp = self.odeblock.odefunc.net(net_in)
                qvec = torch.nn.functional.softplus(temp[:,:self.number_of_hazards],beta=self.odeblock.odefunc.softplus_beta)
                Q = torch.zeros(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1))
                Q[self.transition_matrix.repeat((x.shape[0],1,1))==1] = qvec.flatten()
                Q[torch.eye(self.trans_dim, self.trans_dim,device=x.device).repeat((x.shape[0],1,1)) == 1] = -torch.sum(Q,2).flatten()
                Qvec[i,:,:,:] = Q
        return Qvec
    
    def predict_cumhazard(self,x,tvec):
        with torch.no_grad():
            tvec = tvec.float().to(x.device)
            tvec = torch.unique(torch.cat([torch.tensor([0.],device=x.device),tvec]))
            
            pz0_mean = self.prior_mean(x)
            pz0_logvar = self.prior_var(x)
            epsilon = torch.randn(pz0_mean.size()).to(x.device)
            z0prior = epsilon * torch.exp(.5 * pz0_logvar) + pz0_mean
            
            out = self.odeblock(z0prior,tvec)
            qvec = out[:,:,(2*self.num_probs):(2*self.num_probs+self.number_of_hazards)]
            Qvec = torch.zeros((tvec.shape[0],x.shape[0],self.trans_dim,self.trans_dim),device=x.device)
            Qvec[self.transition_matrix.repeat((tvec.shape[0],x.shape[0],1,1))==1] = qvec.flatten()
        return Qvec
    
    def sample_latent(self,x):
        with torch.no_grad():
            pz0_mean = self.prior_mean(x)
            pz0_logvar = self.prior_var(x)
            epsilon = torch.randn(pz0_mean.size()).to(x.device)
            z0 = epsilon * torch.exp(.5 * pz0_logvar) + pz0_mean
        return x,z0


def loss(odesurv,x,Tstart,Tstop,From,To,trans,status,mu=1e-4,beta=1.,inner_samples=1, outer_samples=1):
    """
        Loss function with parameters:
            - mu: strength of Lyapunov loss
            - beta: parameter in ELBO that controls influence of KL divergence
            - inner_samples: batch gets multiplied by this amount and the average taken
            - outer_samples: loop over # of outer_samples and take average of loss
    """
    trans_exist = torch.tensor([odesurv.transition_matrix[From[i]-1,To[i]-1] for i in range(len(From))])
    trans_exist = torch.where(trans_exist==1)
    x = x[trans_exist]
    Tstart = Tstart[trans_exist]
    Tstop = Tstop[trans_exist]
    From = From[trans_exist]
    To = To[trans_exist]
    status = status[trans_exist]
    
    x_rep = x.repeat(inner_samples,1)
    Tstart_rep = Tstart.repeat(inner_samples)
    Tstop_rep = Tstop.repeat(inner_samples)
    From_rep = From.repeat(inner_samples)
    To_rep = To.repeat(inner_samples)
    status_rep = status.repeat(inner_samples)
    
    lossval = 0.
    regular = 0.
    loglike = 0.
    kull = 0.
    for i in range(outer_samples):
        kl,S,lam,all_h_T = odesurv(x_rep,Tstart_rep,Tstop_rep,From_rep,To_rep,status_rep)
        loglik = -(status_rep*torch.log(lam)+torch.log(S)).mean()
        augmented = all_h_T[:,odesurv.number_of_hazards:]
        reg = torch.norm(augmented,2,dim=1).mean()/odesurv.odeblock.odefunc.num_latent
        kl = kl.mean()
        lossval += (loglik + beta*kl + mu*reg)
        regular += reg
        kull += kl
        loglike += loglik
    return lossval/outer_samples, loglike/outer_samples, kull/outer_samples, regular/outer_samples


def sample_probs(odesurv,x,initial, points=500, inner_samples=1, outer_samples=1, multiplier=1.):
    """
        sample probabilites from 0 to multiplier at "points" number of points from initial state "initial" (e.g. [1,0,0] in the illness-death case starting out at Health)
    """    
    xrep = x.repeat(inner_samples,1)
    curve_vec = []
    for _ in range(outer_samples):
        surv_ode = odesurv.predict(xrep,torch.from_numpy(np.linspace(0,multiplier,points)).float().to(x.device))
        pvec = torch.einsum("ilkj,k->ilj",(surv_ode[:,:,:,:],initial))[:,:,0]
        samp_curves = np.array(pvec.cpu())
        samp_curves = samp_curves.transpose().reshape((int(xrep.shape[0]/x.shape[0]),x.shape[0],points))
        curve_vec.append(samp_curves)
    curve_vec = np.concatenate(curve_vec, axis=0)
    curve_mean = np.mean(curve_vec,axis=0)
    curve_upper = np.percentile(curve_vec,95,axis=0)
    curve_lower = np.percentile(curve_vec,5,axis=0)
    return curve_mean, curve_lower ,curve_upper

def sample_cumhaz(odesurv,x,points=500,inner_samples=1, outer_samples=1):
    xrep = x.repeat((inner_samples,1))
    curve_vec = []
    for _ in range(outer_samples):
        Qvec = odesurv.predict_cumhazard(xrep,torch.from_numpy(np.linspace(0,1,points)).float())
        Qvec = Qvec[Qvec!=0].reshape((Qvec.shape[0]-1,Qvec.shape[1],odesurv.number_of_hazards))
        samp_curves = torch.cat((torch.zeros((1,Qvec.shape[1],Qvec.shape[2]),device=Qvec.device),Qvec),dim=0).cpu().numpy()
        samp_curves = np.array(samp_curves).reshape((points,xrep.shape[0]))
        samp_curves = samp_curves.transpose().reshape((int(xrep.shape[0]/x.shape[0]),x.shape[0],points))
        curve_vec.append(samp_curves)
    curve_vec = np.concatenate(curve_vec, axis=0)
    curve_mean = np.mean(curve_vec,axis=0)
    curve_upper = np.percentile(curve_vec,95,axis=0)
    curve_lower = np.percentile(curve_vec,5,axis=0)
    return curve_mean, curve_lower, curve_upper

def sample_latent(odesurv,xin,inner_samples=1,outer_samples=1):
    xin = xin.repeat((inner_samples,1))
    x_vec,z_vec = [],[]
    for _ in range(outer_samples):
        x,z0 = odesurv.sample_latent(xin)
        x_vec.append(x.cpu().numpy())
        z_vec.append(z0.cpu().numpy())
    x_vec = np.concatenate(x_vec, axis=0)
    z_vec = np.concatenate(z_vec, axis=0)
    return x, z0