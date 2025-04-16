''' Classification depending on Action-wise Binary Map based on
    25/03, 
    
    1. Import D, generate $\bar{C}$ using different columns
    2. generate Y using D and $\bar{C}$ with noise
    3. Solve C with FISTA/rhFISTA
    4. Compare C with $\bar{C}$
'''

import os
from pickle import TRUE
import sys
import glob
from attr import ib
import ipdb
import numpy as np
from einops import rearrange
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch

from modelZoo.networks import BiSC
from utils import sparsity

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def sortKey(s):
    return int(s.split('/')[-1].split('.')[0].replace('dir_d_',''))

def num2rt(rho_rep, theta_rep,
           rmin=0.001, rmax=1.1,
           tmin=0., tmax=torch.pi):
           # rmin=0.8415, rmax=1.0852
           # tmin=0.1687, tmax=3.1052):
    """ Reparameterize rho and theta in a range
    INPUT:
        rho_rep:   torch.tensor(float), (N), range [-inf, inf]
        theta_rep: torch.tensor(float), (N), range [-inf, inf]
    RETURN:
        rho:   torch.tensor(float), (N), range: (rmin, rmax)
        theta: torch.tensor(float), (N), range: (tmin, tmax)
    NOTE:
        Option 1. General range
            rmin, rmax = 0.001, 1.1
            tmin, tmax = 0., torch.pi
        Option 2. Empirical from CVAR
            rmin, rmax = 0.8415, 1.0852
            tmin, tmax = 0.1687, 3.1052
    """
    rho = rmin + (rmax-rmin) * torch.sigmoid(rho_rep)
    theta = tmin + (tmax-tmin) * torch.sigmoid(theta_rep)

    return rho, theta
    

def rt2num(rho, theta,
           rmin=0.001, rmax=1.1,
           tmin=0., tmax=torch.pi):
           # rmin=0.8415, rmax=1.0852
           # tmin=0.1687, tmax=3.1052):
    """ Inversed reparameterization
    INPUT:
        rho:   torch.tensor(float), (N), range: (rmin, rmax)
        theta: torch.tensor(float), (N), range: (tmin, tmax)
    RETURN:
        rho_rep:   torch.tensor(float), (N), range: [-inf, inf]
        theta_rep: torch.tensor(float), (N), range: [-inf, inf]
    """
    rho_rep = (rho-rmin)/(rmax-rmin)
    theta_rep = (theta-tmin)/(tmax-tmin)
    # Avoid log(0) issues
    eps = 1e-6
    rho_rep = torch.clamp(rho_rep, eps, 1-eps)
    theta_rep = torch.clamp(theta_rep, eps, 1-eps)

    return torch.log(rho_rep/(1-rho_rep)), torch.log(theta_rep/(1-theta_rep))


def num2rt_clamp(rho_rep, theta_rep,
           rmin=0.001, rmax=1.1,
           tmin=0., tmax=torch.pi):
           # rmin=0.8415, rmax=1.0852
           # tmin=0.1687, tmax=3.1052):
    """ Reparameterize rho and theta in a range
    INPUT:
        rho_rep:   torch.tensor(float), (N), range [-inf, inf]
        theta_rep: torch.tensor(float), (N), range [-inf, inf]
    RETURN:
        rho:   torch.tensor(float), (N), range: (rmin, rmax)
        theta: torch.tensor(float), (N), range: (tmin, tmax)
    """
    # clamp
    rho = rmin + (rmax-rmin) * torch.clamp(rho_rep, 0, 1)
    theta = tmin + (tmax-tmin) * torch.clamp(theta_rep, 0, 1)

    return rho, theta
    

def rt2num_clamp(rho, theta,
           rmin=0.001, rmax=1.1,
           tmin=0., tmax=torch.pi):
           # rmin=0.8415, rmax=1.0852
           # tmin=0.1687, tmax=3.1052):
    """ Inversed reparameterization
    INPUT:
        rho:   torch.tensor(float), (N), range: (rmin, rmax)
        theta: torch.tensor(float), (N), range: (tmin, tmax)
    RETURN:
        rho_rep:   torch.tensor(float), (N), range: [-inf, inf]
        theta_rep: torch.tensor(float), (N), range: [-inf, inf]
    """
    rho_rep = (rho-rmin)/(rmax-rmin)
    theta_rep = (theta-tmin)/(tmax-tmin)

    return rho_rep, theta_rep


def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='SBAR-PRE')
    parser.add_argument('--modelRoot', default='exps_should_be_saved_on_HDD',
                        help='the work folder for storing experiment results')
    parser.add_argument('--path_list', default='./', help='')
    parser.add_argument('--cus_n', default='', help='customized name')
    parser.add_argument('--mode', default='dy+bi')
    parser.add_argument('--wiRH', default='1', type=str2bool, help='Use Reweighted Heuristic Algorithm')
    parser.add_argument('--wiBI', default='1', type=str2bool, help='Use Binary Code as Input for Classifier')
    parser.add_argument('--pret', default='', help='')
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Multi', help='')
    parser.add_argument('--nClip', default=6, type=int, help='') # sampling=='multi' or sampling!='Single'

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--r_r', default='0.8,1.1', help='rho range')
    parser.add_argument('--r_t', default='0,pi', help='theta range')
    # parser.add_argument('--r_r', default='0.8415,1.0852', help='rho range')
    # parser.add_argument('--r_t', default='0.1687,3.1052', help='theta range')
    parser.add_argument('--g_th', default=0.505, type=float) # 0.501/0.503/0.510
    parser.add_argument('--g_te', default=0.1, type=float)

    parser.add_argument('--gpu_id', default=7, type=int, help='')
    parser.add_argument('--bs', default=32, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    parser.add_argument('--Alpha', default=1e+2, type=float, help='bi loss')
    parser.add_argument('--AlphaP', default=1e+2, type=float, help='pre loss')
    parser.add_argument('--lam2', default=1e+3, type=float, help='mse loss')
    parser.add_argument('--lr_2', default=1e-4, type=float)
    parser.add_argument('--ms', default='20,40', help='milestone for learning rate scheduler')
    parser.add_argument('--Epoch', default=50, type=int, help='')
    
    parser.add_argument('--cls', default='l2', help='')


    return parser


def main(args):
    # Log
    os.system('date')
    # Configurations
    args.bs_t = args.bs
    args.ms = [int(mile) for mile in args.ms.split(',')]
    str_conf = f"LossR_B_{'wiRH' if args.wiRH else 'woRH'}_{'wiBI' if args.wiBI else 'woBI'}"
    args.mode = 'dy+bi' if args.wiBI else 'dy'
    print(f" {args.mode} | {str_conf} | Batch Size: Train {args.bs} | Test {args.bs_t} | Sample {args.sampling}(nClip-{args.nClip})")
    print(f"\t lam_f: {args.lam_f} | r_r: {args.r_r} | r_t: {args.r_t} | g_th: {args.g_th} | g_te: {args.g_te}")
    print(f"\t Alpha: {args.Alpha} | lam2: {args.lam2} | lr_2: {args.lr_2}(milestone: {args.ms})")
    ## Network
    net = BiSC(args).cuda(args.gpu_id)
    sd = torch.load(args.pret, map_location=f"cuda:{args.gpu_id}")['state_dict']
    net.sparseCoding.update_D(sd['sparseCoding.rho'], sd['sparseCoding.theta'])
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Samples LIST !!!'

    # 1. Import D([36,161]), generate $\hat{C}$([3,161,10]) using different rows([1,2,3,4],[1,5,6,7])
    # 2. generate Y,[36,10], using D and $\hat{C}$ with noise, $ \hat{Y}=D\hat{C}+\delta$, $\delta \in N(\mu,\sigma)$ 
    # 3. Solve C with FISTA/rhFISTA
    # 4. Compare C with $\hat{C}$

    # Step 1: Create D tensor
    D = net.sparseCoding.D.to('cpu')
    # D = torch.randn(36, 161)  # Randomly initialized tensor

    # Step 2: Create sparse C tensor
    C_hat = torch.zeros(3, 161, 10)  # Initialize with zeros

    # Populate C with random sparse values
    ## NOTE: for conjugated poles, there is no connection 
    for i in range(C_hat.shape[0]):
        for j in range(C_hat.shape[2]):
            num_nonzero = torch.randint(5, 11, (1,))  # Choose a random number of nonzero elements
            indices = torch.randint(0, 161, (num_nonzero,))  # Random indices
            values = torch.randn(num_nonzero)  # Random values
            C_hat[i, indices, j] = values  # Assign values to random positions
    # ipdb.set_trace()
    ## ipdb> torch.sum(C_hat == 0)/C_hat.numel()
    ## tensor(0.9536)
    
    # Step 3: Compute Y = DC
    Y_hat = D @ C_hat  # Resulting shape is [3, 36, 10]
    # Step 4: Y = DC + noise
    Y_noisy = Y_hat + torch.randn_like(Y_hat) * 0.1 # random numbers from a normal distribution with mean 0 and variance 1
    Y_hat.cuda(args.gpu_id)
    Y_noisy.cuda(args.gpu_id)

    net.eval()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    dict_vis = {'log lamf':[],
                'woRH ML2':[],'woRH ML2 noisy':[],
                'woRH ML1':[],'woRH ML1 noisy':[],
                'woRH sp':[], 'woRH sp noisy':[],
                'woRH xnor':[],'woRH xnor noisy':[],
                'wiRH ML2':[],'wiRH ML2 noisy':[],
                'wiRH ML1':[],'wiRH ML1 noisy':[],
                'wiRH sp':[], 'wiRH sp noisy':[],
                'wiRH xnor':[],'wiRH xnor noisy':[]}
    with torch.no_grad():
        # list_lamf = [1e+1, 1e+0, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4]
        list_lamf = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4]
        
        for lam_f in list_lamf:
            net.sparseCoding.lam_f = lam_f
            dict_vis['log lamf'].append(np.log10(lam_f))

            net.sparseCoding.wiRH = False
            C, D_hat, R = net.sparseCoding(Y_hat)
            C_noisy, D_noisy, R_noisy = net.sparseCoding(Y_noisy)
            # ipdb.set_trace()
            print(f"wiRH: {net.sparseCoding.wiRH} lam_f: {net.sparseCoding.lam_f}")
            print(f"| MS  | Y_hat: {((Y_hat)**2).mean()} Y_noisy: {((Y_noisy)**2).mean()} ")

            dict_vis['woRH ML2'].append(((R-Y_hat)**2).mean().numpy())
            dict_vis['woRH ML2 noisy'].append(((R_noisy-Y_noisy)**2).mean().numpy())
            print(f"| ML2 | R: {dict_vis['woRH ML2'][-1]} R_noisy: {dict_vis['woRH ML2 noisy'][-1]}")
            # axes[0].scatter(log_lamf,((R-Y_hat)**2).mean().numpy(),color='red',marker='*', label='woRH ML2')
            # axes[0].scatter(log_lamf,((R_noisy-Y_noisy)**2).mean().numpy(),color='purple',marker='o', label='woRH ML2 noisy')
            dict_vis['woRH ML1'].append((C_hat.abs()).mean().numpy())
            dict_vis['woRH ML1 noisy'].append((C_noisy.abs()).mean().numpy())
            print(f"| ML1 | C: {dict_vis['woRH ML1'][-1]} C_noisy: {dict_vis['woRH ML1 noisy'][-1]}")
            # axes[1].scatter(log_lamf,(C_hat.abs()).mean().numpy(),color='red',marker='*', label='woRH ML1')
            # axes[1].scatter(log_lamf,(C_noisy.abs()).mean().numpy(),color='purple',marker='o', label='woRH ML1 noisy')
            dict_vis['woRH sp'].append(((torch.sum(C==0)/C.numel()).numpy()))
            dict_vis['woRH sp noisy'].append((torch.sum(C_noisy==0)/C.numel()).numpy())
            print(f"| sp  | C: {dict_vis['woRH sp'][-1]} C_noisy: {dict_vis['woRH sp noisy'][-1]}")
            # axes[2].scatter(log_lamf,(torch.sum(C==0)/C.numel()).numpy(),color='red',marker='*', label='woRH sp')
            # axes[2].scatter(log_lamf,(torch.sum(C_noisy==0)/C.numel()).numpy(),color='purple',marker='o', label='woRH sp noisy')
            nz_hat, nz_ori, nz_noisy = C_hat==0, C==0, C_noisy==0
            dict_vis['woRH xnor'].append((torch.sum(nz_ori==nz_hat)/C_hat.numel()).numpy())
            dict_vis['woRH xnor noisy'].append((torch.sum(nz_noisy==nz_hat)/C_hat.numel()).numpy())
            print(f"| XNOR_hat  | C: {dict_vis['woRH xnor'][-1]} C_noisy: {dict_vis['woRH xnor noisy'][-1]}")
            # axes[3].scatter(log_lamf,(torch.sum(nz_ori==nz_hat)/C_hat.numel()).numpy(),color='red',marker='*', label='woRH xnor')
            # axes[3].scatter(log_lamf,(torch.sum(nz_noisy==nz_hat)/C_hat.numel()).numpy(),color='purple',marker='o', label='woRH xnor noisy')

        
            net.sparseCoding.wiRH = True
            C, _, R = net.sparseCoding(Y_hat)
            C_noisy, _, R_noisy = net.sparseCoding(Y_noisy)
            # ipdb.set_trace()
            print(f"wiRH: {net.sparseCoding.wiRH} lam_f: {net.sparseCoding.lam_f}")
            print(f"| MS | Y_hat: {((Y_hat)**2).mean()} Y_noisy: {((Y_noisy)**2).mean()} ")
            dict_vis['wiRH ML2'].append(((R-Y_hat)**2).mean().numpy())
            dict_vis['wiRH ML2 noisy'].append(((R_noisy-Y_noisy)**2).mean().numpy())
            print(f"| ML2 | R: {dict_vis['wiRH ML2'][-1]} R_noisy: {dict_vis['wiRH ML2 noisy'][-1]}")
            # axes[0].scatter(log_lamf,((R-Y_hat)**2).mean().numpy(),color='blue',marker='+', label='wiRH ML2')
            # axes[0].scatter(log_lamf,((R_noisy-Y_noisy)**2).mean().numpy(),color='green',marker='x', label='wiRH ML2 noisy')
            dict_vis['wiRH ML1'].append((C_hat.abs()).mean().numpy())
            dict_vis['wiRH ML1 noisy'].append((C_noisy.abs()).mean().numpy())
            print(f"| ML1 | C: {(C_hat.abs()).mean()} C_noisy: {(C_noisy.abs()).mean()}")
            # axes[1].scatter(log_lamf,(C_hat.abs()).mean().numpy(),color='blue',marker='+', label='wiRH ML1')
            # axes[1].scatter(log_lamf,(C_noisy.abs()).mean().numpy(),color='green',marker='x', label='wiRH ML1 noisy')
            dict_vis['wiRH sp'].append((torch.sum(C==0)/C.numel()).numpy())
            dict_vis['wiRH sp noisy'].append((torch.sum(C_noisy==0)/C.numel()).numpy())
            print(f"| sp  | C: {torch.sum(C==0)/C.numel()} C_noisy: {torch.sum(C_noisy==0)/C.numel()}")
            # axes[2].scatter(log_lamf,(torch.sum(C==0)/C.numel()).numpy(),color='blue',marker='+', label='wiRH sp')
            # axes[2].scatter(log_lamf,(torch.sum(C_noisy==0)/C.numel()).numpy(),color='green',marker='x', label='wiRH sp noisy')
            nz_hat, nz_ori, nz_noisy = C_hat==0, C==0, C_noisy==0
            dict_vis['wiRH xnor'].append((torch.sum(nz_ori==nz_hat)/C_hat.numel()).numpy())
            dict_vis['wiRH xnor noisy'].append((torch.sum(nz_noisy==nz_hat)/C_hat.numel()).numpy())
            print(f"| XNOR_hat  | C: {torch.sum(nz_ori==nz_hat)/C_hat.numel()} C_noisy: {torch.sum(nz_noisy==nz_hat)/C_hat.numel()}")
            # axes[3].scatter(log_lamf,(torch.sum(nz_ori==nz_hat)/C_hat.numel()).numpy(),color='blue',marker='+', label='wiRH xnor')
            # axes[3].scatter(log_lamf,(torch.sum(nz_noisy==nz_hat)/C_hat.numel()).numpy(),color='green',marker='x', label='wiRH xnor noisy')


    axes[0].scatter(dict_vis['log lamf'],dict_vis['woRH ML2'],color='red',marker='*', label='woRH ML2')
    axes[0].scatter(dict_vis['log lamf'],dict_vis['woRH ML2 noisy'],color='purple',marker='o', label='woRH ML2 noisy')
    axes[0].scatter(dict_vis['log lamf'],dict_vis['wiRH ML2'],color='blue',marker='+', label='wiRH ML2')
    axes[0].scatter(dict_vis['log lamf'],dict_vis['wiRH ML2 noisy'],color='green',marker='x', label='wiRH ML2 noisy')
    axes[0].set_title("ML2")
    axes[0].legend()
    axes[1].scatter(dict_vis['log lamf'],dict_vis['woRH ML1'],color='red',marker='*', label='woRH ML1')
    axes[1].scatter(dict_vis['log lamf'],dict_vis['woRH ML1 noisy'],color='purple',marker='o', label='woRH ML1 noisy')
    axes[1].scatter(dict_vis['log lamf'],dict_vis['wiRH ML1'],color='blue',marker='+', label='wiRH ML1')
    axes[1].scatter(dict_vis['log lamf'],dict_vis['wiRH ML1 noisy'],color='green',marker='x', label='wiRH ML1 noisy')
    axes[1].set_title("ML1")
    axes[1].legend()
    axes[2].scatter(dict_vis['log lamf'],dict_vis['woRH sp'],color='red',marker='*', label='woRH sp')
    axes[2].scatter(dict_vis['log lamf'],dict_vis['woRH sp noisy'],color='purple',marker='o', label='woRH sp noisy')
    axes[2].scatter(dict_vis['log lamf'],dict_vis['wiRH sp'],color='blue',marker='+', label='wiRH sp')
    axes[2].scatter(dict_vis['log lamf'],dict_vis['wiRH sp noisy'],color='green',marker='x', label='wiRH sp noisy')
    axes[2].set_title("Sparsity")
    axes[2].legend()
    axes[3].scatter(dict_vis['log lamf'],dict_vis['woRH xnor'],color='red',marker='*', label='woRH xnor')
    axes[3].scatter(dict_vis['log lamf'],dict_vis['woRH xnor noisy'],color='purple',marker='o', label='woRH xnor noisy')
    axes[3].scatter(dict_vis['log lamf'],dict_vis['wiRH xnor'],color='blue',marker='+', label='wiRH xnor')
    axes[3].scatter(dict_vis['log lamf'],dict_vis['wiRH xnor noisy'],color='green',marker='x', label='wiRH xnor noisy')
    axes[3].set_title("XNOR")
    axes[3].legend()
    plt.tight_layout()
    plt.savefig('test-1.png')
    plt.close()
        
    #     pass

    # # NOTE: Norlmalize Y along column(UNFINISHED YET)
    # # Compute mean and standard deviation along dimension b (dim=1)
    # mean_hat = Y_hat.mean(dim=1, keepdim=True)  # Shape (3, 1, 10)
    # std_hat = Y_hat.std(dim=1, keepdim=True) + 1e-8  # Shape (3, 1, 10)
    # Y_hat_norm = (Y_hat - mean_hat) / std_hat
    # mean_noisy = Y_noisy.mean(dim=1, keepdim=True)  # Shape (3, 1, 10)
    # std_noisy = Y_noisy.std(dim=1, keepdim=True) + 1e-8  # Shape (3, 1, 10)
    # Y_noisy_norm = (Y_noisy - mean_noisy) / std_noisy

    # max_hat = Y_hat.min(dim=1, keepdim=True)[0]  # Shape (3, 1, 10)
    # min_hat = Y_hat.max(dim=1, keepdim=True)[0]  # Shape (3, 1, 10)
    # Y_hat_norm = (Y_hat - min_hat) / (max_hat - min_hat + 1e-8)
    # max_noisy = Y_noisy.min(dim=1, keepdim=True)[0]  # Shape (3, 1, 10)
    # min_noisy = Y_noisy.max(dim=1, keepdim=True)[0]  # Shape (3, 1, 10)
    # Y_noisy_norm = (Y_noisy - min_noisy) / (max_noisy - min_noisy + 1e-8)

    # net.eval()
    # with torch.no_grad():
    #     C, _, R = net.sparseCoding(Y_hat_norm)
    #     C_noisy, _, R_noisy = net.sparseCoding(Y_noisy_norm)
    #     # ipdb.set_trace()
    #     print(f"| MSE | Y_hat: {((Y_hat)**2).mean()} Y_noisy: {((Y_noisy)**2).mean()} ")
    #     print(f"| ML2 | R: {((R-Y_hat)**2).mean()} R_noisy: {((R_noisy-Y_noisy)**2).mean()}")
    #     print(f"| ML1 | C: {(C_hat.abs()).mean()} C_noisy: {(C_noisy.abs()).mean()}")
    #     print(f"| sp  | C: {torch.sum(C==0)/C.numel()} C_noisy: {torch.sum(C_noisy==0)/C.numel()}")
    #     nz_hat, nz_ori, nz_noisy = C_hat==0, C==0, C_noisy==0
    #     print(f"| XNOR_hat  | C: {torch.sum(nz_ori==nz_hat)/C_hat.numel()} C_noisy: {torch.sum(nz_noisy==nz_hat)/C_hat.numel()}")

        
    #     pass





if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    main(args)