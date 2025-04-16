# DIR-1: train dictionary, N-UCLA dataset
import os
import ipdb
import time
import numpy as np
from einops import rearrange
import random
import argparse
import datetime
# from ptflops import get_model_complexity_info
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# from dataset.crossView_UCLA import NUCLA_CrossView
from dataset.crossView_UCLA_ske import NUCLA_CrossView
from modelZoo.networks import BiSC
from utils import sparsity

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='SBAR')
    parser.add_argument('--modelRoot', default='exps_should_be_saved_on_HDD',
                        help='the work folder for storing experiment results')
    parser.add_argument('--path_list', default='./', help='')
    parser.add_argument('--cus_n', default='', help='customized name')
    parser.add_argument('--mode', default='dy+bi')
    parser.add_argument('--wiRH', default='1', type=str2bool, help='Use Reweighted Heuristic Algorithm')
    parser.add_argument('--wiBI', default='1', type=str2bool, help='Use Binary Code as Input for Classifier')
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Single', help='')
    parser.add_argument('--nClip', default=6, type=int, help='') # sampling=='multi' or sampling!='Single'

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--r_r', default='0.8,1.1', help='rho range')
    parser.add_argument('--r_t', default='0,pi', help='theta range')
    # parser.add_argument('--r_r', default='0.8415,1.0852', help='rho range')
    # parser.add_argument('--r_t', default='0.1687,3.1052', help='theta range')
    parser.add_argument('--g_th', default=0.502, type=float) # 0.501/0.503/0.510
    parser.add_argument('--g_te', default=0.1, type=float)

    parser.add_argument('--gpu_id', default=7, type=int, help='')
    parser.add_argument('--bs', default=32, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    parser.add_argument('--Alpha', default=1e+2, type=float, help='bi loss')
    parser.add_argument('--lam2', default=1e+3, type=float, help='mse loss')
    parser.add_argument('--lr_2', default=1e-3, type=float)
    parser.add_argument('--ms', default='20,40', help='milestone for learning rate scheduler')
    parser.add_argument('--Epoch', default=50, type=int, help='')

    return parser


def test(dl, net, epoch):
    net.eval()
    with torch.no_grad():
        lossVal_t = []
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
        Sp_0, Sp_th = [], []
        for _, sample in enumerate(dl):
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            # -> batch_size x num_clips, t, num_joint x dim_joint
            input_skeletons = rearrange(skeletons, 'n c t j d -> (n c) t (j d)')
            C, R_C, B, R_B = net(input_skeletons, True)
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            # ipdb.set_trace()
            ML1_C_B_col = (C*B).abs().sum(dim=1).mean()
            loss_t = args.lam2 * MSE_B + args.Alpha * ML1_C_B_col

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
    args.saveModel = os.path.join(args.modelRoot,
                                  f"NUCLA_CV_{args.setup}_{args.sampling}/DIR_D_{str_conf}/")
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    ## Network
    net = BiSC(args).cuda(args.gpu_id)
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Samples LIST !!!'
    path_list = args.path_list + f"/data/CV/{args.setup}/"
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               setup=args.setup, dataType=args.dataType,
                               sampling=args.sampling, nClip=args.nClip,
                               T=args.T, maskType='None')
    trainloader = DataLoader(trainSet, shuffle=True,
                             batch_size=args.bs, num_workers=args.nw)
    testSet = NUCLA_CrossView(root_list=path_list, phase='test',
                              setup=args.setup, dataType=args.dataType,
                              sampling=args.sampling, nClip=args.nClip,
                              T=args.T, maskType='None')
    testloader = DataLoader(testSet, shuffle=False,
                            batch_size=args.bs_t, num_workers=args.nw)
    # Training Strategy
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), 
                                lr=args.lr_2, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.ms, gamma=0.1) # 30,50
    
    test(trainloader, net, 0)
    test(testloader, net, 0)
    torch.save({'epoch': 0, 
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()}, 
                args.saveModel +'dir_d_'+ str(0) + '.pth')
    
    for epoch in range(1, args.Epoch+1):
        print('training epoch:', epoch)
        net.train()

        lossVal = []
        l1_C, l1_C_B, l1_B, mseC, mseB = [], [], [], [], []
        start_time = time.time()
        for _, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            # -> batch_size x num_clips, t, num_joint x dim_joint
            input_skeletons = rearrange(skeletons, 'n c t j d -> (n c) t (j d)')
            ### output from model
            C, R_C, B, R_B = net(input_skeletons, False)
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean() 
            ML1_C_B = (C*B).abs().mean() # c: batch_size(32) x num_clips(6), num_poles(161), dim_data(50)
            # ipdb.set_trace()
            ML1_C_B_col = (C*B).abs().sum(dim=1).mean()
            loss = args.lam2 * MSE_B + args.Alpha * ML1_C_B_col
            
            #### BP and Log
            loss.backward()
            optimizer.step()
            l1_C.append(ML1_C.detach().cpu())
            l1_C_B.append(ML1_C_B.detach().cpu())
            mseC.append(MSE_C.detach().cpu())
            l1_B.append(ML1_B.detach().cpu())
            mseB.append(MSE_B.detach().cpu())
            lossVal.append(loss.detach().cpu())
        # Log
        print('Train epoch:', epoch,
              'loss:', np.mean(np.asarray(lossVal)),
              'L1_C:', np.mean(np.asarray(l1_C)),
              'L1_C_B:', np.mean(np.asarray(l1_C_B)),
              'mseC:', np.mean(np.asarray(mseC)),
              'L1_B:', np.mean(np.asarray(l1_B)),
              'mseB:', np.mean(np.asarray(mseB)),
              f'duration:{(time.time() - start_time) / 60:.4f} min')

        if epoch % 1 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, 
                        args.saveModel +'dir_d_'+ str(epoch) + '.pth')
            test(testloader, net, epoch)
        scheduler.step()
    
    os.system('date')
    print('END\n')


if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    main(args)
