import torch
import torch.nn as nn
# from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import DataLoader, TensorDataset
from ray import tune

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from pycox.evaluation import EvalSurv

random_seed = 1337# 137
torch.manual_seed(random_seed)
np.random.seed(random_seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print(device)

# Early stopping class from https://github.com/Bjarten/early-stopping-pytorch
from EarlyStopping import EarlyStopping
from SurvNODE_alt import *

def measures(odesurv,initial,x,Tstart,Tstop,From,To,trans,status, multiplier=1.,points=500):
    with torch.no_grad():
        time_grid = np.linspace(0, multiplier, points)
        pvec = torch.zeros((points,x.shape[0]))
        surv_ode = odesurv.predict(x,torch.from_numpy(np.linspace(0,multiplier,points)).float().to(x.device))
        pvec = torch.einsum("ilkj,k->ilj",(surv_ode[:,:,:,:],initial))[:,:,0].cpu()
        pvec = np.array(pvec.cpu().detach())
        surv_ode_df = pd.DataFrame(pvec)
        surv_ode_df.loc[:,"time"] = np.linspace(0,multiplier,points)
        surv_ode_df = surv_ode_df.set_index(["time"])
        ev_ode = EvalSurv(surv_ode_df, np.array(Tstop.cpu()), np.array(status.cpu()), censor_surv='km')
        conc = ev_ode.concordance_td('antolini')
        ibs = ev_ode.integrated_brier_score(time_grid)
        inbll = ev_ode.integrated_nbll(time_grid)
    return conc,ibs,inbll

from pycox import datasets
from sklearn_pandas import DataFrameMapper
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_dataloader(df,Tmax,batchsize, shuffle=True):
    cols_standardize = ['x0','x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
    cols_leave = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)
    X = x_mapper.fit_transform(df).astype('float32')
    
    X = torch.from_numpy(X).float().to(device)
    T = torch.from_numpy(df[["duration"]].values).float().flatten().to(device)
    Tmax = torch.tensor(Tmax).to(device)
    T = T/Tmax
    T[T==0] = 1e-8
    E = torch.from_numpy(df[["event"]].values).float().flatten().to(device)

    Tstart = torch.from_numpy(np.array([0 for i in range(T.shape[0])])).float().to(device)
    From = torch.tensor([1],device=device).repeat((T.shape))
    To = torch.tensor([2],device=device).repeat((T.shape))
    trans = torch.tensor([1],device=device).repeat((T.shape))

    dataset = TensorDataset(X,Tstart,T,From,To,trans,E)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return loader

def odesurv_manual_benchmark(df_train, df_test,config):
    torch.cuda.empty_cache()
    df_val = df_train.sample(frac=config["traintest_fraction"])
    df_train = df_train.drop(df_val.index)
    
    Tmax = df_train["duration"].max()
    
    train_loader = make_dataloader(df_train,Tmax/config["multiplier"],config["batch_size"])
    val_loader = make_dataloader(df_val,Tmax/config["multiplier"],len(df_val))
    test_loader = make_dataloader(df_test,Tmax/config["multiplier"],len(df_test), shuffle=False)
    
    num_in = 14
    num_latent = config["num_latent"]
    layers_encoder =  [config["encoder_neurons"]]*config["num_encoder_layers"]
    dropout_encoder = [config["encoder_dropout"]]*config["num_encoder_layers"]
    layers_odefunc =  [config["odefunc_neurons"]]*config["num_odefunc_layers"]
    dropout_odefunc = []

    trans_matrix = torch.tensor([[np.nan,1],[np.nan,np.nan]]).to(device)

    encoder = Encoder(num_in,num_latent,layers_encoder, dropout_encoder).to(device)
    odefunc = ODEFunc(trans_matrix,num_in,num_latent,layers_odefunc).to(device)
    # odefunc = ODEFunc(trans_matrix,num_in,num_latent,layers_odefunc,dropout_odefunc).to(device)
    block = ODEBlock(odefunc).to(device)
    odesurv = SurvNODE(block,encoder).to(device)

    optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = config["weight_decay1"], lr=config["lr_stage1"])
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["scheduler_epochs"], gamma=config["scheduler_factor"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["scheduler_factor"], patience=config["scheduler_epochs"], verbose="True")
    
    early_stopping = EarlyStopping("test",patience=config["patience"], verbose=True)
    t = trange(1000, desc='Training', leave=True, ncols=120)
    for i in t:
        odesurv.train()
        for mini,ds in enumerate(train_loader):
            myloss,_,_ = loss(odesurv,*ds,mu=config["mu"]) #
            optimizer.zero_grad()
            myloss.backward()    
            optimizer.step()
        
        odesurv.eval()
        with torch.no_grad():
            lossval,loglike,conc,ibs,ibnll = 0., 0., 0., 0., 0.
            for _,ds in enumerate(val_loader):
                t1,_,_ = loss(odesurv,*ds,mu=config["mu"])
                lossval += t1.item()
                t1,t2,t3 = measures(odesurv,torch.tensor([1.,0.],device=device),*ds,multiplier=config["multiplier"])
                conc += t1
                ibs += t2
                ibnll += t3
            # conc_test,ibs_test,ibnll_test = 0., 0., 0.
            # for _,ds in enumerate(test_loader):
            #     t1,t2,t3 = measures(odesurv,torch.tensor([1.,0.],device=device),*ds,multiplier=config["multiplier"])
            #     conc_test += t1
            #     ibs_test += t2
            #     ibnll_test += t3
            early_stopping(lossval/len(val_loader), odesurv)
            print("Scheduler")
            scheduler.step(lossval/len(val_loader))
            # print("lossval: "+str(lossval/len(val_loader))+" c: "+str(conc/len(val_loader))+" ibs: "+str(ibs/len(val_loader))+" ibnll: "+str(ibnll/len(val_loader)))
            t.refresh()
            t.set_postfix({"lossval.:": lossval/len(val_loader),
                           "c" : conc/len(val_loader),
                           "ibs" : ibs/len(val_loader),
                           "ibll": -ibnll/len(val_loader)})

        temp_t = torch.from_numpy(np.linspace(0.,1,100))
        prediction_hazard = []
        durr = []
        with torch.no_grad():
            for df in test_loader:
                # print(df)
                # print(len(temp_t))
                # print(df[0].shape)
                out = odesurv.predict_cumhazard(df[0],temp_t).cpu()
                prediction_hazard.append(out)
                durr = df[2]
                # print(df[0][:,2])
        prediction_hazard = torch.cat(prediction_hazard,dim=1)

        plt.figure()
        plt.xlim(0, Tmax)
        plt.ylim(0, 16)
        colors = ["red", "cyan", "midnightblue", "magenta"]
        for patient in range(4):
            plt.plot((temp_t*Tmax).numpy(),prediction_hazard[:,patient,0,1].numpy(), color=colors[patient])
            plt.vlines(x = (Tmax*durr[patient]).cpu().numpy(), ymin=0, ymax=0.4, linewidth=2, color=colors[patient])
        # plt.text(2, 5, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
        plt.savefig("step_{}".format(i))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    odesurv.load_state_dict(torch.load('test_checkpoint.pt'))

    # optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = config["weight_decay2"], lr=config["lr_stage2"])
    # early_stopping = EarlyStopping("Checkpoints/surv",patience=config["patience"], verbose=True)
    # for i in tqdm(range(500)):
    #     odesurv.train()
    #     for mini,ds in enumerate(train_loader):
    #         myloss,t2,_ = loss(odesurv,*ds)
    #         optimizer.zero_grad()
    #         myloss.backward()    
    #         optimizer.step()
            
    #     odesurv.eval()
    #     with torch.no_grad():
    #         lossval = 0
    #         for _,ds in enumerate(val_loader):
    #             t1,t2,_ = loss(odesurv,*ds)
    #             lossval += t1.item()
        
    # #     scheduler.step()
    #     early_stopping(lossval/len(val_loader), odesurv)
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break
    
    odesurv.eval()
    with torch.no_grad():
        conc,ibs,ibnll = 0., 0., 0.
        for ds in test_loader:
            t1,t2,t3 = measures(odesurv,torch.tensor([1.,0.],device=device),*ds,multiplier=config["multiplier"])
            conc += t1
            ibs += t2
            ibnll += t3

    return conc/len(test_loader), ibs/len(test_loader), ibnll/len(test_loader)

from sklearn.model_selection import KFold
from pycox import datasets

kfold = KFold(5,shuffle=True)
df_all = datasets.support.read_df()
# print(df_all[0:13].head())

gen = kfold.split(df_all)

# config = {
#     "lr": 1e-3,
#     "weight_decay": 1e-4,
#     "num_latent": 100,
#     "encoder_neurons": 20,
#     "num_encoder_layers": 1,
#     "encoder_dropout": 0.2,
#     "odefunc_neurons": 500, 
#     "num_odefunc_layers": 2,
#     "batch_size": 512,
#     "multiplier": 1.,
#     "mu": 1e-4,
#     "patience": 20,
#     "softplus_beta": 1.,
#     "scheduler_epochs": 10,
#     "scheduler_factor": 0.2,
#     "traintest_fraction": 0.2
# }

config = {
    "lr_stage1": 1e-4,
    "lr_stage2": 1e-4,
    "weight_decay1": 1e-5,
    "weight_decay2": 1e-7,
    "num_latent": 200,
    "encoder_neurons": 400,
    "num_encoder_layers": 2,
    "encoder_dropout": 0,
    "odefunc_neurons": 500, 
    "num_odefunc_layers": 3,
    "batch_size": 512,
    "multiplier": 1.,
    "mu": 1e-4,
    "patience": 10,
    "softplus_beta": 0.1,
    "scheduler_epochs": 10,
    "scheduler_factor": 0.2,
    "traintest_fraction": 0.2
}

# config = {
#     "lr_stage1": tune.grid_search([1e-5,5e-5,1e-4]),
#     "weight_decay1": tune.grid_search([1e-7,1e-5,1e-3]),
#     "num_latent": 200,
#     "encoder_neurons": 400,
#     "num_encoder_layers": 2,
#     "encoder_dropout": 0.,
#     "odefunc_neurons": tune.grid_search([400,1000]),
#     "num_odefunc_layers": tune.grid_search([2,4]),
#     "batch_size": 512,
#     "multiplier": 1.,
#     "mu": 1e-4,
#     "patience": 10,
#     "softplus_beta": 1.,
#     "scheduler_epoch": 20,
#     "scheduler_gamma": 0.1,
#      "traintest_fraction": 0.2    
# }


odesurv_bench_vals = []
for g in gen:
    df_train = df_all.iloc[g[0]]
    df_test =  df_all.iloc[g[1]]
    conc, ibs, ibnll = odesurv_manual_benchmark(df_train,df_test,config)
    odesurv_bench_vals.append([conc,ibs,ibnll])


print("Results:")

scores = torch.tensor(odesurv_bench_vals)
print(scores)
print(torch.mean(scores, dim=0))
print(torch.std(scores, dim=0))

