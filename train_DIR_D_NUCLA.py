# DIR-1: train dictionary, N-UCLA dataset
from ast import parse
import os
import ipdb
import time
import numpy as np
import random
import argparse
import datetime
# from ptflops import get_model_complexity_info
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset.crossView_UCLA import NUCLA_CrossView
from modelZoo.BinaryCoding import DyanEncoder, binarizeSparseCode
from utils import gridRing
# from test_cls_CV import testing, getPlots

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='CVARDIF')
    parser.add_argument('--modelRoot', default='/data/Dan/202111_CVAR/202410_CVARDIF/DIR_D',
                        help='the work folder for storing experiment results')
    parser.add_argument('--cus_n', default='', help='customized name')
    
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--sampling', default='Single', help='')

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80*2, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--gumbel_thresh', default=0.1, type=float)

    parser.add_argument('--gpu_id', default=7, type=int, help='')
    parser.add_argument('--bs', default=2, type=int, help='')
    parser.add_argument('--nw', default=1, type=int, help='')
    parser.add_argument('--Epoch', default=100, type=int, help='')
    parser.add_argument('--lr', default=1e-3, type=float)

    return parser

def main(args):
    # Configurations
    ## Paths
    args.saveModel = args.modelRoot + f'/{args.sampling}_bi_DIR_D/'
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    ## Dictionary
    P, Pall = gridRing(args.N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    ## Network
    ### DYAN: FISTA
    # net = DyanEncoder(Drr, Dtheta,
    #                   args.lam_f,
    #                   args.gpu_id)
    ### DYAN: FISTA, ReWeighted Heuristic Algorithm, Binarization Module
    net = binarizeSparseCode(num_binary=128, Drr=Drr, Dtheta=Dtheta,
                             gpu_id=args.gpu_id, Inference=False, 
                             fistaLam=args.lam_f)
    net.cuda(args.gpu_id)
    path_list = f'/home/dan/ws/202209_CrossView/202409_CVAR_yuexi_lambdaS/data/CV/{args.setup}/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               dataType='2D',  T=args.T, setup=args.setup,
                               sampling=args.sampling, maskType='score',
                               nClip=10)
    trainloader = DataLoader(trainSet, shuffle=True,
                             batch_size=args.bs, num_workers=args.nw)
    testSet = NUCLA_CrossView(root_list=path_list, phase='test',
                              dataType='2D', T=args.T, setup=args.setup,
                              sampling=args.sampling, maskType='score',
                              nClip=10)
    testloader = DataLoader(testSet, shuffle=True,
                            batch_size=args.bs, num_workers=args.nw)
    # Training Strategy
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), 
                                lr=args.lr, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    # Loss
    mseLoss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss(reduction='mean')

    net.train()
    # Loss = []
    for epoch in range(0, args.Epoch+1):
        print('training epoch:', epoch)
        lossVal = []
        lossMSE = []
        lossL1 = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            t = skeletons.shape[1] # (batch_size x num_clips) x t x dim_joint? x num_joint?
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1) # (batch_size x num_clips) x t x (dim_joint x num_joint)
            ### rh-dyan + bi
            binaryCode, output_skeletons, _ = net(input_skeletons, t, 0.5)
            target_coeff = torch.zeros_like(binaryCode).cuda(args.gpu_id)
            loss = mseLoss(output_skeletons, input_skeletons) + 0.5*l1Loss(binaryCode,target_coeff)
            #### BP and Log
            loss.backward()
            optimizer.step()
            # ipdb.set_trace()
            lossMSE.append(mseLoss(output_skeletons, input_skeletons).data.item())
            lossL1.append(0.5*l1Loss(binaryCode,target_coeff).data.item())
            lossVal.append(loss.data.item())
        end_time = time.time()
        # print('epoch:', epoch, 'loss:', np.mean(np.asarray(lossVal)), 'time(h):', (end_time - start_time) / 3600)
        print('epoch:', epoch, 'mse loss:', np.mean(np.asarray(lossMSE)), 'L1 loss:', np.mean(np.asarray(lossL1)),
              'duration:', (end_time - start_time) / 3600, 'hr')

        if epoch % 5 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.saveModel + str(epoch) + '.pth')
            with torch.no_grad():
                ERROR = torch.zeros(testSet.__len__(), 1)

                for i, sample in enumerate(testloader):
                    skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
                    t = skeletons.shape[1]
                    input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)
                    # 'regular dyan'
                    # _,_,output_skeletons = net.forward2(input_skeletons, t) # reconst
                    # output_skeletons = net.prediction(input_skeletons[:,0:t-1], t-1)

                    # 'rhDyan+Bi'
                    _, output_skeletons,_ = net(input_skeletons, t, 0.5)
                    error = torch.norm(output_skeletons - input_skeletons).cpu()
                    ERROR[i] = error

                print('epoch:', epoch, 'error:', torch.mean(ERROR))

        scheduler.step()
    print('done')

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    main(args)
