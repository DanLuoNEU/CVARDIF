''' Visualize the Binary Code
    25/02, Dan
'''
import os
import sys
import glob
from attr import ib
import ipdb
import numpy as np
from einops import rearrange
import random
import imageio
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    
    parser.add_argument('--vis', default='sum_B_grid', help='')


    return parser


def vis_sum(dl, net, epoch):
    hist_B = torch.zeros(10, args.nClip, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
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


def vis_sum_d(dl, net, epoch):
    hist_B = torch.zeros(10, args.nClip, 161).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
                hist_B[gt[i_clip//args.nClip], i_clip%args.nClip,:] += torch.sum(B[i_clip],dim=1)
                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        # Move to CPU and convert to NumPy
        hist_B_np = hist_B.cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/vis_B_sum_d'
        os.makedirs(output_dir, exist_ok=True)

        # Loop over the indices: i (10), j (6), and k (50)
        for i in range(hist_B_np.shape[0]):  # Loop over dimension 0 (10)
            for j in range(hist_B_np.shape[1]):  # Loop over dimension 1 (6)
                # Extract the vector along dim=2 (length 161) for the given indices
                data = hist_B_np[i, j, :]
                # Create a new figure for each bar plot
                plt.figure()
                
                # The x-axis will be the indices from 0 to 160
                x = np.arange(data.shape[0])
                
                # Create a bar plot
                plt.bar(x, data, alpha=0.7, color='blue')
                plt.title(f'Histogram for index (a-{i}, c-{j})')
                plt.xlabel('index-B')
                plt.ylabel('sum')
                
                # Save the figure locally
                filename = os.path.join(output_dir, f'hist_a{i}_c{j}.png')
                plt.savefig(filename)
                plt.close()  # Close the figure to free memory


def vis_sum_cd(dl, net, epoch):
    hist_B = torch.zeros(10, 161).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
                hist_B[gt[i_clip//args.nClip], :] += torch.sum(B[i_clip],dim=1)
                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        # Move to CPU and convert to NumPy
        hist_B_np = hist_B.cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/vis_B_sum_cd'
        os.makedirs(output_dir, exist_ok=True)

        # Loop over the indices: i (10), j (6), and k (50)
        for i in range(hist_B_np.shape[0]):  # Loop over dimension 0 (10)
                # Extract the vector along dim=2 (length 161) for the given indices
                data = hist_B_np[i, :]
                # Create a new figure for each bar plot
                plt.figure()
                
                # The x-axis will be the indices from 0 to 160
                x = np.arange(data.shape[0])
                
                # Create a bar plot
                plt.bar(x, data, alpha=0.7, color='blue')
                plt.title(f'Histogram for B (a-{i})')
                plt.xlabel('index-B')
                plt.ylabel('sum')
                
                # Save the figure locally
                filename = os.path.join(output_dir, f'hist_a{i}.png')
                plt.savefig(filename)
                plt.close()  # Close the figure to free memory


def log_multi(dl, net, epoch):
    hist_B = torch.zeros(10, 161).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
        Sp_0, Sp_th = [], []
        lines_info_clips = []
        T_stat = []
        for _, sample in enumerate(dl):
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            gt = sample['action'].cuda(args.gpu_id)
            view = sample['cam']
            name_sample = sample['sample_name']
            ids_sample = sample['input_skeletons']['ids_sample'] # list, len=args.nClip, ids_sample[0].shape=args.bs
            T_sample = sample['input_skeletons']['T_sample']
            for i_b, txt_v in enumerate(view):
                list_ids = [int(ids_clip[i_b]) for ids_clip in ids_sample]
                lines_info_clips.append(txt_v+f' {name_sample[i_b]} {T_sample[i_b]} {list_ids}\n')
                T_stat.append(int(T_sample[i_b]))
                # ipdb.set_trace()
            # ipdb.set_trace()
            # -> batch_size x num_clips, t, num_joint x dim_joint
            input_skeletons = rearrange(skeletons, 'n c t j d -> (n c) t (j d)')
            C, R_C, B, R_B = net(input_skeletons, True)
            # record B
            for i_clip in range(B.shape[0]):
                hist_B[gt[i_clip//args.nClip], :] += torch.sum(B[i_clip],dim=1)
                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        with open("multi_clip_ids_sample.txt", "w") as file:
            file.write(f'T mean: {np.mean(np.asarray(T_stat))}\n')
            file.writelines(lines_info_clips)
            

def vis_sum_grid(dl, net, epoch):
    hist_B = torch.zeros(10, args.nClip, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        # Move to CPU and convert to NumPy
        hist_B_np = hist_B.cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/sum_B_Grid'
        os.makedirs(output_dir, exist_ok=True)

        # Loop over the indices: i (10), j (6), and k (50)
        for i in range(hist_B_np.shape[1]):  # Loop over dimension 0 (10)
            
            # plt.figure()
            fig, axes = plt.subplots(2, 5, figsize=(40, 40))
            
            for j in range(hist_B_np.shape[0]):
                data = hist_B_np[j,i]
                d_min = data.min()
                d_max = data.max()
                d_norm = (data - d_min) / (d_max - d_min)

                ax = axes[j//5, j%5]  # Handle single-subplot case
                ax.imshow(d_norm, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"a{j}-c{i}")
                ax.axis('off')
            
            plt.tight_layout()
            # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
            # Save the figure locally
            filename = os.path.join(output_dir, f'Grid_B_c{i}.png')
            plt.savefig(filename, bbox_inches='tight',dpi=300)
            plt.close()  # Close the figure to free memory


def vis_sum_c_grid(dl, net, epoch):
    hist_B = torch.zeros(10, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
                hist_B[gt[i_clip//args.nClip],:,:] += B[i_clip]
                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        # Move to CPU and convert to NumPy
        hist_B_np = (hist_B>0).int().cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/sum_B_c_Grid'
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 5, figsize=(40, 40))
        # Loop over the indices: i (10)
        for i in range(hist_B_np.shape[0]):  # Loop over dimension 0 (10)
            data = hist_B_np[i]
            # Draw Binary Heatmap
            ax = axes[i//5, i%5]  # Handle single-subplot case
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"a{i}")
            ax.axis('off')
            
        plt.tight_layout()
        # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
        # Save the figure locally
        filename = os.path.join(output_dir, f'Grid_B_c.png')
        plt.savefig(filename, bbox_inches='tight',dpi=300)
        plt.close()  # Close the figure to free memory


def vis_sum_c_grid10x10(dl, net, epoch):
    hist_B = torch.zeros(10, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
                hist_B[gt[i_clip//args.nClip],:,:] += B[i_clip]
                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        # Turn counting B to used poles binary map
        # ipdb.set_trace()
        B_A = hist_B>0 # num_cls, num_poles, dim_data 
        mat_cnt_xnor = torch.zeros(B_A.shape[0],B_A.shape[0])
        for i in range(B_A.shape[0]):
            for j in range(B_A.shape[0]):
                mat_cnt_xnor[i,j] = (~(B_A[i] ^ B_A[j])).sum()

        # Move to CPU and convert to NumPy
        mat_cnt_xnor = mat_cnt_xnor.cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/sum_B_c_Grid10x10'
        os.makedirs(output_dir, exist_ok=True)
        # Normalize data to range [0, 1] for colormap
        norm = mcolors.Normalize(vmin=np.min(mat_cnt_xnor), vmax=np.max(mat_cnt_xnor))

        # Plot heatmap
        plt.figure()
        # plt.imshow(mat_cnt_xnor, cmap='viridis', norm=norm)

        plt.imshow(mat_cnt_xnor, cmap='viridis', aspect="auto")
        plt.colorbar(label="cnt_xnor")
        plt.title("Grid 10x10")
            
        plt.tight_layout()
        # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
        # Save the figure locally
        filename = os.path.join(output_dir, f'Grid_B_c_10x10.png')
        plt.savefig(filename, bbox_inches='tight',dpi=300)
        plt.close()  # Close the figure to free memory


def vis_sum_grid10x6x6(dl, net, epoch):
    hist_B = torch.zeros(10, args.nClip, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')
        
        # Turn counting B to used poles binary map
        # ipdb.set_trace()
        B_A = hist_B>0 # num_cls, num_poles, dim_data 
        mat_cnt_xnor = torch.zeros(B_A.shape[0], B_A.shape[1], B_A.shape[1])
        for i_a in range(B_A.shape[0]):
            for i in range(B_A.shape[1]):
                for j in range(B_A.shape[1]):
                    mat_cnt_xnor[i_a, i, j] = (~(B_A[i_a,i] ^ B_A[i_a, j])).sum()

        # Move to CPU and convert to NumPy
        mat_cnt_xnor = mat_cnt_xnor.cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/sum_B_Grid10x6x6'
        os.makedirs(output_dir, exist_ok=True)

        # Plot heatmap
        fig, axes = plt.subplots(2, 5, figsize=(40, 10))
        # Loop over the indices: i (10)
        for i in range(mat_cnt_xnor.shape[0]):  # Loop over dimension 0 (10)
            data = mat_cnt_xnor[i]
            # Draw Binary Heatmap
            ax = axes[i//5, i%5]  # Handle single-subplot case
            im = ax.imshow(data, cmap='viridis', aspect="auto")
            fig.colorbar(im, ax=ax, label="cnt_xnor")
            ax.set_title(f"a{i}")
            ax.axis('off')
        plt.tight_layout()
        # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
        # Save the figure locally
        filename = os.path.join(output_dir, f'Grid_B_10x6x6.png')
        plt.savefig(filename, bbox_inches='tight',dpi=300)
        plt.close()  # Close the figure to free memory


def vis_sum_grid10x5center(dl, net, epoch):
    hist_B = torch.zeros(10, args.nClip, 161, 50).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data
    cnt_act = [0 for _ in range(10)]
    B_samples = torch.zeros(hist_B.shape[0], 5, hist_B.shape[2], hist_B.shape[3]).cuda(args.gpu_id) # act, num_clips, num_poles, dim_data

    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
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
                if i_clip%args.nClip==2 and cnt_act[gt[i_clip//args.nClip]]<5:
                    B_samples[gt[i_clip//args.nClip],cnt_act[gt[i_clip//args.nClip]]] = B[i_clip]
                    cnt_act[gt[i_clip//args.nClip]] += 1

                # ipdb.set_trace()
            
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B

            lossVal_t.append(loss_t.detach().cpu())
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
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
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')

        # Move to CPU and convert to NumPy
        hist_B = (hist_B>0).cpu().numpy()
        B_xnor = torch.zeros_like(B_samples).cpu().numpy()
        B_samples = (B_samples>0).cpu().numpy()
        # Create an output directory to save figures
        output_dir = os.path.dirname(args.pret)+'/sum_c2_B_Grid10x5'
        os.makedirs(output_dir, exist_ok=True)

        # Plot heatmap
        for i_a in range(B_samples.shape[0]):
            fig, axes = plt.subplots(1, 5, figsize=(20, 10))
            for i_s in range(B_samples.shape[1]): 
                data = (~( B_samples[i_a,i_s] ^ hist_B[i_a,2]))
                # Draw Binary Heatmap
                ax = axes[i_s]  # Handle single-subplot case
                im = ax.imshow(data, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"a{i_a}_s{i_s}")
                ax.axis('off')
            plt.tight_layout()
            # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
            # Save the figure locally
            filename = os.path.join(output_dir, f"{args.pret.split('/')[-1].split('.')[0]}_c2_Grid10x5_a{i_a}.png")
            plt.savefig(filename, bbox_inches='tight',dpi=300)
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
    print(f"\t Alpha: {args.Alpha} | lam2: {args.lam2} | lr_2: {args.lr_2}(milestone: {args.ms})")
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

    if args.vis == 'sum':           vis_sum(trainloader, net, 0)
    elif args.vis == 'sum_d':       vis_sum_d(trainloader, net, 0)
    elif args.vis == 'sum_cd':      vis_sum_cd(trainloader, net, 0)
    elif args.vis == 'sum_grid':           vis_sum_grid(trainloader, net, 0)
    # elif args.vis == 'sum_grid':         vis_sum_a_grid(trainloader, net, 0)
    elif args.vis == 'sum_c_grid':         vis_sum_c_grid(trainloader, net, 0)
    elif args.vis == 'sum_c_grid10x10':    vis_sum_c_grid10x10(trainloader, net, 0)
    elif args.vis == 'clip':               log_multi(trainloader, net, 0)
    elif args.vis == 'sum_grid10x6x6':     vis_sum_grid10x6x6(trainloader, net, 0)
    elif args.vis == 'sum_grid10x5':       vis_sum_grid10x5center(trainloader, net, 0)
    

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    main(args)