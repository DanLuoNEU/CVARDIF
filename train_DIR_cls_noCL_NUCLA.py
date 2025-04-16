# DIR-2: train CVAR, N-UCLA dataset
import os
import ipdb
import time
from einops import rearrange, reduce
import argparse
from matplotlib import pyplot as plt
# from ptflops import get_model_complexity_info

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.crossView_UCLA_ske import np, random, NUCLA_CrossView
from modelZoo.networks import nn,CVARDIF
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
    
    parser = argparse.ArgumentParser(description='CVARDIF')
    parser.add_argument('--modelRoot', default='exps_should_be_saved_on_HDD',
                        help='the work folder for storing experiment results')
    parser.add_argument('--path_list', default='./', help='')
    parser.add_argument('--pretrain',  default='', help='')
    parser.add_argument('--cus_n', default='', help='customized name')
    
    parser.add_argument('--dataset', default='NUCLA', help='')
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--num_class', default=10, type=int, help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Single', help='')
    parser.add_argument('--nClip', default=6, type=int, help='') # sampling=='multi' or sampling!='Single'
    parser.add_argument('--bs', default=32, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    
    parser.add_argument('--mode', default='dy+bi+cl', help='dy+bi+cl | dy+cl | rgb+dy')
    parser.add_argument('--wiRH', default='1', type=str2bool, help='')
    parser.add_argument('--wiBI', default='1', type=str2bool, help='Use Binary Code as Input for Classifier')
    parser.add_argument('--withMask', default='0', type=str2bool, help='')
    parser.add_argument('--maskType', default='None', help='')
    parser.add_argument('--fusion', default='0', type=str2bool, help='')
    parser.add_argument('--groupLasso', default='0', type=str2bool, help='')

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80, type=int, help='')
    parser.add_argument('--lam_f', default=0.01, type=float)
    parser.add_argument('--g_th', default=0.505, type=float) # 0.503
    parser.add_argument('--g_te', default=0.1, type=float)

    parser.add_argument('--gpu_id', default=0, type=int, help='')
    parser.add_argument('--Epoch', default=100, type=int, help='')
    parser.add_argument('--lr', default=1e-4, type=float, help='sparse coding')
    parser.add_argument('--lr_2', default=1e-4, type=float, help='classifier')
    parser.add_argument('--ms', default='25,40', help='milestone for learning rate scheduler')
    parser.add_argument('--Alpha', default=0.1, type=float, help='bi loss')
    parser.add_argument('--lam1', default=1, type=float, help='cls loss')
    parser.add_argument('--lam2', default=0.5, type=float, help='mse loss')

    return parser


def test(dataloader, net, epoch):
    
    net.eval()
    with torch.no_grad():
        count, pred_cnt = 0,0
        l1_C_t, l1_C_B_t, l1_B_t, mseC_t, mseB_t = [], [], [], [], []
        Sp_0, Sp_th = [], []
        for i, sample in enumerate(dataloader):
            # print('sample:', i)
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            gt = sample['action'].cuda(args.gpu_id)

            Nsample, nClip, t = skeletons.shape[0], skeletons.shape[1], skeletons.shape[2]
            input_skeletons =rearrange(skeletons, 'b c t n d -> (b c) t (n d)')

            # input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
            actPred, _, C, R_C, B, R_B = net(input_skeletons)

            actPred = actPred.reshape(Nsample, nClip, actPred.shape[-1])
            actPred = torch.mean(actPred, 1)
            pred = torch.argmax(actPred, 1)

            correct = torch.eq(gt, pred).int()
            count += gt.shape[0]
            pred_cnt += torch.sum(correct).data.item()

            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            l1_C_t.append(ML1_C.detach().cpu())
            l1_C_B_t.append(ML1_C_B.detach().cpu())
            mseC_t.append(MSE_C.detach().cpu())
            l1_B_t.append(ML1_B.detach().cpu())
            mseB_t.append(MSE_B.detach().cpu())
            sp_0,sp_th =sparsity(C)
            Sp_0.append(sp_0.detach().cpu())
            Sp_th.append(sp_th.detach().cpu())

        Acc = pred_cnt/count
        print('Test epoch:', epoch,
            'Acc:', f'{Acc*100:.4f}%',
            'L1_C:',   np.mean(np.asarray(l1_C_t)),
            'L1_C_B:', np.mean(np.asarray(l1_C_B_t)),
            'mseC:',   np.mean(np.asarray(mseC_t)),
            'L1_B:',   np.mean(np.asarray(l1_B_t)),
            'mseB:',   np.mean(np.asarray(mseB_t)),
            f'Sp_0:{np.mean(np.asarray(Sp_0))}', 
            f'Sp_th:{np.mean(np.asarray(Sp_th))}')

    return Acc

def main(args):
    str_conf = f"{'wiRH' if args.wiRH else 'woRH'}_{'wiBI' if args.wiBI else 'woBI'}"
    args.mode = 'dy+bi+cl' if args.wiBI else 'dy+cl'
    print(f" {args.mode} | {str_conf} | Batch Size: Train {args.bs} | Test {args.bs} ")
    print(f"\tlam_f: {args.lam_f} | g_th: {args.g_th} | g_te: {args.g_te}")
    print('Experiment config | setup:',args.setup,'sampling:', args.sampling,
          '\n\tAlpha(bi):',args.Alpha,'lam1(cls):',args.lam1,'lam2(mse):',args.lam2,
          'lr(mse):',args.lr,'lr_2(cls):',args.lr_2)
    args.saveModel = os.path.join(args.modelRoot,
                                  f'NUCLA_CV_{args.setup}_{args.sampling}/DIR_cls_noCL_{str_conf}/')
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    '============================================= Main Body of script================================================='
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Sample LIST !!!'
    path_list = args.path_list + f"/data/CV/{args.setup}/"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               setup=args.setup, dataType=args.dataType,
                               sampling=args.sampling, nClip=args.nClip,
                               T=args.T, maskType=args.maskType) 
    trainloader = DataLoader(trainSet, shuffle=True,
                             batch_size=args.bs, num_workers=args.nw)
    testSet = NUCLA_CrossView(root_list=path_list, phase='test',
                              setup=args.setup, dataType=args.dataType,
                              sampling=args.sampling, nClip=args.nClip,
                              T=args.T, maskType=args.maskType)
    testloader = DataLoader(testSet, shuffle=False,
                            batch_size=args.bs, num_workers=args.nw)
    # Pretrained Dictionary
    assert args.pretrain!='', '!!! NO Pretrained Dictionary !!!'
    print('pretrain:', args.pretrain)
    stateDict = torch.load(args.pretrain, map_location=args.map_loc)['state_dict']
    # Model
    net = CVARDIF(args.N, args.T, args.lam_f,
                    args.wiRH,
                    args.wiBI, args.g_th, args.g_te,
                    args.num_class, args.dataType, 
                    useCL=False).cuda(args.gpu_id)
    net.sparseCoding.update_D(stateDict['sparseCoding.rho'],stateDict['sparseCoding.theta'])
    # Parameters Count
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters (including non-trainable): {total_params}")
    # Freeze the Dictionary part
    net.train()
    net.sparseCoding.rho.requires_grad = False
    net.sparseCoding.theta.requires_grad = False
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                                lr=args.lr_2, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)# 25,40
    Criterion = torch.nn.CrossEntropyLoss()
    mseLoss = torch.nn.MSELoss()
    L1loss = torch.nn.SmoothL1Loss()
    test(trainloader, net, 0)
    test(testloader, net, 0)

    ACC, acc_max = [], (0,0)
    for epoch in range(1, args.Epoch+1):
        print('start training epoch:', epoch)
        # net.train()
        lossVal, lossCls = [], []
        # lossBi, lossMSE_C, lossMSE_B = [], [], []
        l1_C, l1_C_B, l1_B, mseC, mseB = [], [], [], [], []
        start_time = time.time()
        for _, sample in enumerate(trainloader):
            optimizer.zero_grad()
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            gt_label = sample['action'].cuda(args.gpu_id)

            Nsample, nClip, t = skeletons.shape[0], skeletons.shape[1], skeletons.shape[2]
            input_skeletons =rearrange(skeletons, 'b c t n d -> (b c) t (n d)')
            
            keep_index = None
            actPred, _, C, R_C, B, R_B = net(input_skeletons)
            # actPred = reduce(actPred, 'a_b c -> a c', 'mean', a='a', b='b')
            actPred = rearrange(actPred,'(b c) a -> b c a', c=nClip)
            actPred = reduce(actPred, 'b c a -> b a', 'mean')

            # actPred = actPred.reshape(Nsample, nClip, args.num_class)
            # actPred = torch.mean(actPred, 1)
            
            loss = args.lam1*Criterion(actPred, gt_label)
            # bi_gt = torch.zeros_like(B).cuda(args.gpu_id)
            # lossMSE_C.append(mseLoss(R_C, input_skeletons).data.item())
            # lossMSE_B.append(mseLoss(R_B, input_skeletons).data.item())
            # lossBi.append(L1loss(B, bi_gt).data.item())

            loss.backward()
            # ipdb.set_trace()
            optimizer.step()
            MSE_C = ((R_C - input_skeletons)**2).mean()
            MSE_B = ((R_B - input_skeletons)**2).mean()
            ML1_C = C.abs().mean()
            ML1_B = B.abs().mean()
            ML1_C_B = (C*B).abs().mean()
            l1_C.append(ML1_C.detach().cpu())
            l1_C_B.append(ML1_C_B.detach().cpu())
            mseC.append(MSE_C.detach().cpu())
            l1_B.append(ML1_B.detach().cpu())
            mseB.append(MSE_B.detach().cpu())

            lossVal.append(loss.data.item())
            lossCls.append(Criterion(actPred, gt_label).data.item())
        # loss_val = torch.mean(torch.tensor(lossVal))
        # print(f'epoch: {epoch}, |loss: {loss_val} |cls:{torch.mean(torch.tensor(lossCls))}', 
        #       f'\n\t|mse_b: {torch.mean(torch.tensor(lossMSE_B))} |mse_c: {torch.mean(torch.tensor(lossMSE_C))} |bi: {torch.mean(torch.tensor(lossBi))}')
        end_time = time.time()
        print('Train epoch:', epoch,
              'loss:',   np.mean(np.asarray(lossVal)),
              'cls',     np.mean(np.asarray(lossCls)),
              'L1_C:',   np.mean(np.asarray(l1_C)),
              'L1_C_B:', np.mean(np.asarray(l1_C_B)),
              'mseC:',   np.mean(np.asarray(mseC)),
              'L1_B:',   np.mean(np.asarray(l1_B)),
              'mseB:',   np.mean(np.asarray(mseB)),
              'training time(min):', (end_time - start_time)/60)
        scheduler.step()
        if epoch % 2 == 0:
            net.eval()
            Acc = test(testloader, net, epoch)
            if Acc >= acc_max[1]:
                acc_max = (epoch, Acc)
                torch.save({'state_dict': net.state_dict(),
                   'optimizer': optimizer.state_dict()}, 
                   args.saveModel + f"dir_cls_noCL_{epoch}.pth")
            print(f'  Acc MAX: {acc_max[1]*100:.4f}%(ep@{acc_max[0]})')
            ACC.append(Acc)
        
    torch.cuda.empty_cache()
    print('done')

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)
    
    main(args)