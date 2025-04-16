''' Visualize the Binary Code
    25/02, Dan
'''
import os
import sys
import glob
import ipdb
import numpy as np
from einops import rearrange
import random
import imageio
import argparse
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset.crossView_UCLA_ske import NUCLA_CrossView
from modelZoo.networks import BiSC
from utils import sparsity

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


seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
    parser.add_argument('--PRE', default=1, type=int, help='')
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

    return parser


def test(dl, net, epoch):
    hist_B = torch.zeros(10, args.nClip, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t, mseBR_t, mseBP_t  = [], [], [], [], [], [], []
        Sp_0, Sp_th = [], []
        for _, sample in enumerate(dl):
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            gt = sample['action'].cuda(args.gpu_id)
            # ipdb.set_trace()
            # -> batch_size x num_clips, t, num_joint x dim_joint
            input_skeletons = rearrange(skeletons, 'n c t j d -> (n c) t (j d)')
            C, R_C, B, R_B = net(input_skeletons, True)
            # record B
            for i_clip in range(B.shape[0]):
                hist_B[gt[i_clip//args.nClip], i_clip%args.nClip,:,:] += B[i_clip]
                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            MSE_B_R = ((R_B[:,:args.T-args.PRE,:] - input_skeletons[:,:args.T-args.PRE,:])**2).mean()
            MSE_B_PRE = ((R_B[:,-args.PRE:,:] - input_skeletons[:,-args.PRE:,:])**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B_R + args.AlphaP * MSE_B_PRE + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
            mseBR_t.append(MSE_B_R.detach().cpu())
            mseBP_t.append(MSE_B_PRE.detach().cpu())
            sp_0,sp_th =sparsity(C)
            Sp_0.append(sp_0.detach().cpu())
            Sp_th.append(sp_th.detach().cpu())
        # log
        print('Test epoch:', epoch, 
            'loss:', np.mean(np.asarray(lossVal_t)),
            'L1_C:', np.mean(np.asarray(l1_C_t)),
            'L1_C_B:', np.mean(np.asarray(l1_C_B_t)),
            'mseC:', np.mean(np.asarray(mseC_t)),
            'L1_B:', np.mean(np.asarray(l1_B_t)),
            'mseB:', np.mean(np.asarray(mseB_t)),
            f'Sp_0:{np.mean(np.asarray(Sp_0))}', 
            f'Sp_th:{np.mean(np.asarray(Sp_th))}',
            'mseBR:', np.mean(np.asarray(mseBR_t)),
            'mseBPRE:', np.mean(np.asarray(mseBP_t)))
        
        # Move to CPU and convert to NumPy
        hist_B_np = hist_B.cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/vis_B_sum'
        os.makedirs(output_dir, exist_ok=True)

        # Loop over the indices: i (10), j (6), and k (50)
        for i in range(hist_B_np.shape[0]):  # Loop over dimension 0 (10)
            for j in range(hist_B_np.shape[1]):  # Loop over dimension 1 (6)
                for k in range(hist_B_np.shape[3]):  # Loop over dimension 3 (50)
                    # Extract the vector along dim=2 (length 161) for the given indices
                    data = hist_B_np[i, j, :, k]
                    # Create a new figure for each bar plot
                    plt.figure()
                    
                    # The x-axis will be the indices from 0 to 160
                    x = np.arange(data.shape[0])
                    
                    # Create a bar plot
                    plt.bar(x, data, alpha=0.7, color='blue')
                    plt.title(f'Histogram for index (a-{i}, c-{j}, d-{k})')
                    plt.xlabel('index-B')
                    plt.ylabel('sum')
                    
                    # Save the figure locally
                    filename = os.path.join(output_dir, f'hist_a{i}_c{j}_d{k}.png')
                    plt.savefig(filename)
                    plt.close()  # Close the figure to free memory


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
    print(f"\t Alpha: {args.Alpha} | AlphaP: {args.AlphaP} | lam2: {args.lam2} | lr_2: {args.lr_2}(milestone: {args.ms})")
    ## Network
    net = BiSC(args).cuda(args.gpu_id)
    sd = torch.load(args.pret, map_location=f"cuda:{args.gpu_id}")['state_dict']
    net.sparseCoding.update_D(sd['sparseCoding.rho'], sd['sparseCoding.theta'])
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Samples LIST !!!'
    path_list = args.path_list + f"/data/CV/{args.setup}/"
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               setup=args.setup, dataType=args.dataType,
                               sampling=args.sampling, nClip=args.nClip,
                               T=args.T, maskType='None')
    trainloader = DataLoader(trainSet, shuffle=False,
                             batch_size=args.bs, num_workers=args.nw)

    test(trainloader, net, 0)
    

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    main(args)