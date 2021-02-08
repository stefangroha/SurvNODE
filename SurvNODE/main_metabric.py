import torch
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from sklearn_pandas import DataFrameMapper
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from pycox.evaluation import EvalSurv

random_seed = 137 #1337# 137
torch.manual_seed(random_seed)
np.random.seed(random_seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print(device)

# Early stopping class from https://github.com/Bjarten/early-stopping-pytorch
from EarlyStopping import EarlyStopping
from SurvNODE import *

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
df = datasets.metabric.read_df()
# print(df.head())
df["From"] = 1.
df["To"] = 2.
df["trans"] = 1.
df["Tstart"] = 0.
# df["covar"] = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]].values

# df = df.rename({"duration": "Tstop", "event": "status"}, axis='columns')
df["duration"] = df["duration"]/df["duration"].max()

survdata_test = df.sample(frac=0.2)
survdata_train = df.drop(survdata_test.index)
# survdata_val = survdata_train.sample(frac=0.2)
# survdata_train = survdata_train.drop(survdata_val.index)


# from sklearn.model_selection import train_test_split



# def get_dataset(df,Tmax):
#     # x = torch.from_numpy(np.array(df[["covar"]])).float().to(device)
#     x = torch.from_numpy(np.array(df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]])).float().to(device)
#     Tstart = torch.from_numpy(np.array(df[["Tstart"]])).flatten().float().to(device)
#     Tstop = torch.from_numpy(np.array(df[["Tstop"]])).flatten().float().to(device)
#     Tstart = Tstart/Tmax*multiplier
#     Tstop = Tstop/Tmax*multiplier
#     From = torch.from_numpy(np.array(df[["From"]])).flatten().int().to(device)
#     To = torch.from_numpy(np.array(df[["To"]])).flatten().int().to(device)
#     trans = torch.from_numpy(np.array(df[["trans"]])).flatten().int().to(device)
#     status = torch.from_numpy(np.array(df[["status"]])).flatten().float().to(device)
#     dataset = TensorDataset(x,Tstart,Tstop,From,To,trans,status)
#     return dataset

# multiplier = 1.
# Tmax = max(torch.from_numpy(np.array(df["Tstop"])).flatten().float().to(device))
# # print(Tmax)
# train_loader = DataLoader(get_dataset(survdata_train,Tmax), batch_size=512, shuffle=True)
# val_loader = DataLoader(get_dataset(survdata_val,Tmax), batch_size=512, shuffle=True)
# print(survdata_test.shape[0])
# test_loader = DataLoader(get_dataset(survdata_test,Tmax), batch_size=survdata_test.shape[0], shuffle=False)

# num_in = 9
# num_latent = 10
# layers_encoder = [20]*2#[10]*2
# dropout_encoder = [0.1]*2
# layers_odefunc = [50]*5#[50]*3
# # dropout_odefunc = []

# trans_matrix = torch.tensor([[np.nan,1],[np.nan,np.nan]]).to(device)

# encoder = Encoder(num_in,num_latent,layers_encoder, dropout_encoder).to(device)
# odefunc = ODEFunc(trans_matrix,num_in,num_latent,layers_odefunc).to(device)
# block = ODEBlock(odefunc).to(device)
# odesurv = SurvNODE(block,encoder).to(device)

# optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = 1e-6, lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

# early_stopping = EarlyStopping("Checkpoints/surv",patience=20, verbose=True)
# for i in tqdm(range(500)):
#     odesurv.train()
#     for mini,ds in enumerate(train_loader):
#         # print(ds)
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
        
# odesurv.load_state_dict(torch.load('Checkpoints/surv_checkpoint.pt'))

# optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = 1e-7, lr=1e-4)
# early_stopping = EarlyStopping("Checkpoints/surv",patience=20, verbose=True)
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
        
# odesurv.load_state_dict(torch.load('Checkpoints/surv_checkpoint.pt'))

# for ds in test_loader:
#     print(measures(odesurv, torch.tensor([1., 0.]).cuda(),*ds, multiplier=1.,points=500))



##############################################################################




def make_train_dataloader(df,Tmax,batchsize):

    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    standardize = [([col], scaler) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)
    X = x_mapper.fit_transform(df).astype('float32')
    
    X = torch.from_numpy(X).to(device)
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
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return loader, x_mapper

def make_test_dataloader(df,Tmax,batchsize,x_mapper):

    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    X = x_mapper.transform(df).astype('float32')
    X = torch.from_numpy(X).to(device)
    T = torch.from_numpy(df[["duration"]].values).float().flatten().to(device)
    #Tmax = torch.tensor(Tmax).to(device)
    T = T/Tmax
    T[T==0] = 1e-8
    E = torch.from_numpy(df[["event"]].values).float().flatten().to(device)

    Tstart = torch.from_numpy(np.array([0 for i in range(T.shape[0])])).float().to(device)
    From = torch.tensor([1],device=device).repeat((T.shape))
    To = torch.tensor([2],device=device).repeat((T.shape))
    trans = torch.tensor([1],device=device).repeat((T.shape))

    dataset = TensorDataset(X,Tstart,T,From,To,trans,E)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    return loader

def odesurv_manual_benchmark(df_train, df_test, config):
    torch.cuda.empty_cache()
    df_val = df_train.sample(frac=config["traintest_fraction"])
    df_train = df_train.drop(df_val.index)
    Tmax = df_train["duration"].max()
    
    train_loader, x_mapper = make_train_dataloader(df_train, Tmax/config["multiplier"], config["batch_size"])
    val_loader = make_test_dataloader(df_val, Tmax/config["multiplier"], len(df_val), x_mapper)
    test_loader = make_test_dataloader(df_test, Tmax/config["multiplier"], len(df_test), x_mapper)
    
    num_in = 9
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
    for i in tqdm(range(config["initial"])):
        odesurv.train()
        for mini,ds in enumerate(train_loader):
            myloss,_,_ = loss(odesurv, *ds ,mu=config["mu"]) #
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
            scheduler.step(lossval/len(val_loader))
            print("lossval: "+str(lossval/len(val_loader))+" c: "+str(conc/len(val_loader))+" ibs: "+str(ibs/len(val_loader))+" ibnll: "+str(ibnll/len(val_loader)))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    odesurv.load_state_dict(torch.load('test_checkpoint.pt'))

    optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = config["weight_decay2"], lr=config["lr_stage2"])
    early_stopping = EarlyStopping("Checkpoints/surv",patience=config["patience"], verbose=True)
    for i in tqdm(range(500)):
        odesurv.train()
        for mini,ds in enumerate(train_loader):
            myloss,t2,_ = loss(odesurv,*ds)
            optimizer.zero_grad()
            myloss.backward()    
            optimizer.step()
            
        odesurv.eval()
        with torch.no_grad():
            lossval = 0
            for ds in val_loader:
                t1,t2,_ = loss(odesurv,*ds)
                lossval += t1.item()
        
    #     scheduler.step()
        early_stopping(lossval/len(val_loader), odesurv)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    odesurv.eval()
    with torch.no_grad():
        conc,ibs,ibnll = 0., 0., 0.
        for ds in test_loader:
            t1,t2,t3 = measures(odesurv,torch.tensor([1.,0.],device=device),*ds,multiplier=config["multiplier"])
            conc += t1
            ibs += t2
            ibnll += t3
    return lossval/len(val_loader), conc/len(test_loader), ibs/len(test_loader), ibnll/len(test_loader)

from sklearn.model_selection import KFold
from pycox import datasets

kfold = KFold(5,shuffle=True)
df_all = datasets.metabric.read_df()
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

# config = {
#     "lr_stage1": 1e-3,
#     "lr_stage2": 1e-4,
#     "weight_decay1": 1e-4,
#     "weight_decay2": 1e-7,
#     "num_latent": 30,
#     "encoder_neurons": 20,
#     "num_encoder_layers": 2,
#     "encoder_dropout": 0.2,
#     "odefunc_neurons": 500,#100, 
#     "num_odefunc_layers": 5,#3,
#     "batch_size": 512,
#     "multiplier": 1.,
#     "mu": 1e-4,
#     "patience": 20,
#     "softplus_beta": 1.,
#     "scheduler_epochs": 15,
#     "scheduler_factor": 0.2,
#     "traintest_fraction": 0.2
# }

# config = {
#     "lr_stage1": 1e-3,
#     "lr_stage2": 1e-4,
#     "weight_decay1": 1e-4,
#     "weight_decay2": 1e-7,
#     "num_latent": 70,#30,
#     "encoder_neurons": 20,
#     "num_encoder_layers": 5,
#     "encoder_dropout": 0.,
#     "odefunc_neurons": 500,#100, 
#     "num_odefunc_layers": 5,#3,
#     "batch_size": 512,
#     "multiplier": 1.,
#     "mu": 1e-4,
#     "patience": 20,
#     "softplus_beta": 1.,
#     "scheduler_epochs": 15,
#     "scheduler_factor": 0.2,
#     "traintest_fraction": 0.2
# }

# config = {
#     "initial": 100,
#     "lr_stage1": 5e-4,
#     "lr_stage2": 5e-5, #3e-4]),
#     "weight_decay1": 1e-4,
#     "weight_decay2": 1e-7,
#     "num_latent": 53,#30,
#     "encoder_neurons": 17,
#     "num_encoder_layers": 4,
#     "encoder_dropout": 0.,
#     "odefunc_neurons": 699,#100, 
#     "num_odefunc_layers": 3,
#     "batch_size": 256,
#     "multiplier": 1.,
#     "mu": 1e-4,
#     "patience": 10,
#     "softplus_beta": 1.,
#     "scheduler_epochs": 15,
#     "scheduler_factor": 0.2,
#     "traintest_fraction": 0.2
# }

#hyperopt foung

config = {
    'encoder_neurons': 19, 
    'initial': 12, 
    'num_encoder_layers': 3, 
    'num_latent': 43, 
    'num_odefunc_layers': 3, 
    'odefunc_neurons': 591,
    'batch_size': 512, 
    'encoder_dropout': 0.0, 
    'encoder_neurons': 19, 
    'initial': 12, 
    'lr_stage1': 0.0001, 
    'lr_stage2': 5e-05, 
    'mu': 0.0001, 
    'multiplier': 1.0, 
    'num_encoder_layers': 3, 
    'num_latent': 43, 
    'num_odefunc_layers': 3, 
    'odefunc_neurons': 591, 
    'patience': 20, 
    'scheduler_epochs': 15, 
    'scheduler_factor': 0.2, 
    'softplus_beta': 1.0, 
    'traintest_fraction': 0.2, 
    'weight_decay1': 0.0001, 
    'weight_decay2': 1e-07
}

odesurv_bench_vals = []
for g in gen:
    df_train = df_all.iloc[g[0]]
    df_test =  df_all.iloc[g[1]]
    loss_value, conc, ibs, ibnll = odesurv_manual_benchmark(df_train,df_test,config)
    odesurv_bench_vals.append([conc,ibs,ibnll])


print("Results:")

scores = torch.tensor(odesurv_bench_vals)
print(scores)
print(torch.mean(scores, dim=0))
print(torch.std(scores, dim=0))

from hyperopt import hp
# args = {
#     "lr": hp.choice("lr", [1e-4, 3e-4]),
#     "weight_decay": hp.choice("weight_decay", [1e-3, 1e-5]),
#     "num_latent": hp.randint('num_latent', 20, 35),
#     "encoder_neurons": hp.randint('encoder_neurons', 500, 1500),
#     "num_encoder_layers": 2,
#     "encoder_dropout": 0.1,
#     "odefunc_neurons": 1000,
#     "num_odefunc_layers": 3,
#     "batch_size": 1/3,
#     "multiplier": 3.,
#     "mu": 1e-4,
#     "softplus_beta": 1.,
#     "scheduler_epoch": 50,
#     "scheduler_gamma": 0.1,
#     "patience": 20
# }

args = {
    "initial": hp.randint("initial", 5, 15),
    "lr_stage1": 1e-4,
    "lr_stage2": 5e-5,
    "weight_decay1": 1e-4,
    "weight_decay2": 1e-7,
    "num_latent": hp.randint('num_latent', 20, 100),#30,
    "encoder_neurons": hp.randint('encoder_neurons', 10, 35),
    "num_encoder_layers": hp.randint('num_encoder_layers', 2, 5),
    "encoder_dropout": 0.,
    "odefunc_neurons": hp.randint('odefunc_neurons', 500, 1500),#100, 
    "num_odefunc_layers": hp.randint('num_odefunc_layers', 2, 5),#3,
    "batch_size": 512,
    "multiplier": 1.,
    "mu": 1e-4,
    "patience": 20,
    "softplus_beta": 1.,
    "scheduler_epochs": 15,
    "scheduler_factor": 0.2,
    "traintest_fraction": 0.2
}



# # define an objective function
# def objective(args):
#     # print(args)
#     loss, conc, ibs, ibnll = odesurv_manual_benchmark(survdata_train,survdata_test,args)
#     # print("L, c, ibs, ibnll: {}{}{}{}".format(loss, conc, ibs, ibnll))
#     return loss

# # define a search space

# # minimize the objective over the space
# from hyperopt import fmin, tpe, space_eval
# best = fmin(objective, args, algo=tpe.suggest, max_evals=100)

# print(best)
# print(space_eval(args, best))