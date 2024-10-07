# DIR-2: train CVAR, N-UCLA dataset
import time
import ipdb
import argparse
from matplotlib import pyplot as plt
# from ptflops import get_model_complexity_info

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.crossView_UCLA import np, random, NUCLA_CrossView
from modelZoo.BinaryCoding import nn, gridRing, classificationWSparseCode
from modelZoo.BinaryCoding import classificationWBinarizationRGB, classificationWBinarizationRGBDY
from modelZoo.BinaryCoding import contrastiveNet
from test_cls_CV import testing, getPlots
from utils import load_pretrainedModel_endtoEnd

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
    parser.add_argument('--modelRoot', default='/data/Dan/202111_CVAR/NUCLA/',
                        help='the work folder for storing experiment results')
    parser.add_argument('--cus_n', default='', help='customized name')
    
    parser.add_argument('--dataset', default='NUCLA', help='')
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--num_class', default=10, type=int, help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Single', help='')
    parser.add_argument('--bs', default=4, type=int, help='')
    parser.add_argument('--nw', default=12, type=int, help='')
    
    parser.add_argument('--mode', default='dy+bi+cl', help='dy+bi+cl | dy+cl | rgb+dy')
    parser.add_argument('--RHdyan', default='0', type=str2bool, help='')
    parser.add_argument('--withMask', default='0', type=str2bool, help='')
    parser.add_argument('--maskType', default='None', help='')
    parser.add_argument('--fusion', default='0', type=str2bool, help='')
    parser.add_argument('--groupLasso', default='0', type=str2bool, help='')


    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80*2, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--gumbel_thresh', default=0.503, type=float)

    parser.add_argument('--gpu_id', default=7, type=int, help='')
    parser.add_argument('--Epoch', default=100, type=int, help='')
    parser.add_argument('--lr', default=1e-3, type=float, help='classifier')
    parser.add_argument('--lr_2', default=1e-3, type=float, help='sparse coding')
    parser.add_argument('--Alpha', default=1e-2, type=float, help='bi loss')
    parser.add_argument('--lam1', default=2, type=float, help='cls loss')
    parser.add_argument('--lam2', default=0.5, type=float, help='mse loss')

    return parser

def main(args):
    '------configuration:-------------------------------------------'
    # dataset = 'NUCLA'
    # Alpha = 0.01# bi loss
    # lam1 = 2 # cls loss
    # lam2 = 0.5 # mse loss
    # N = 80*2
    # Epoch = 100
    # num_class = 10
    # dataType = '2D'
    # sampling = 'Single' #sampling strategy

    # RHdyan = False
    # withMask = False
    # groupLasso = False
    # group_lam = 0.1
    # if groupLasso:
    #     fistaLam = 0.00
    # else:
    #     fistaLam = 0.1

    # maskType = 'None'
    # if sampling == 'Single':
    #     num_workers = 4
    #     bz = 12
    # else:
    #     num_workers = 4
    #     bz = 4

    # T = 36 # input clip length
    # # mode = 'dy+bi+cl'
    # # mode = 'dy+cl'
    # mode = 'rgb+dy'
    # setup = 'setup1'
    # modelRoot = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/'
    

    # pdb.set_trace()
    # fusion = False
    # num_class = 10
    # # v1,v2 train, v3 test;
    # lr = 1e-3 # classifier
    # lr_2 = 1e-3  # sparse codeing
    # gumbel_thresh = 0.503 # or try 0.503
    # print('gumbel threshold:',gumbel_thresh)
    # 'change to your own model path'

    '============================================= Main Body of script================================================='

    P,Pall = gridRing(args.N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()


    path_list = f'/home/dan/ws/202209_CrossView/202409_CVAR_yuexi_lambdaS/data/CV/{args.setup}/' # './data/CV/' + args.setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, dataType=args.dataType,
                               sampling=args.sampling,  T=args.T, maskType=args.maskType,
                               setup=args.setup, phase='train', nClip=4)
    trainloader = DataLoader(trainSet, shuffle=True, batch_size=args.bs, num_workers=args.nw)

    testSet = NUCLA_CrossView(root_list=path_list, dataType=args.dataType,
                              sampling=args.sampling, T=args.T, maskType=args.maskType, 
                              setup=args.setup, phase='test', nClip=4)
    testloader = DataLoader(testSet, shuffle=True, batch_size=args.bs, num_workers=args.nw)

    if args.mode == 'dy+cl':
        # NOTE: this mode is only for regular DYAN so far
        # pretrain = './pretrained/NUCLA/' + setup + '/' + sampling + '/pretrainedDyan.pth'
        # pretrain = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/regularDYAN_seed123/60.pth'
        pretrain = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/rhDYAN_bi/100.pth'
        print('pretrain:', pretrain)
        stateDict = torch.load(pretrain, map_location=args.map_loc)['state_dict']
    
        # Drr1 = stateDict['rr'] 
        # Dtheta1 = stateDict['theta']
        Drr = stateDict['sparseCoding.rr'] # for RH
        Dtheta = stateDict['sparseCoding.theta']
        # 'for thresholding dict'
        # keepIndex_all = np.load('./thresholded_dict_rhDyan/rhDyan_setup1_keepIndex.npz', allow_pickle=True)['x'].item()
        # most_freq_index = np.load('./thresholded_dict_rhDyan/rhDyan_setup1_keepIndex.npz', allow_pickle=True)['y']

        # threshs = list(keepIndex_all.keys())
        # keep_index = keepIndex_all[threshs[1]]
        # keep_index = most_freq_index
        # print('using threshold:', threshs[-1])
        # ipdb.set_trace()
        # Drr_new = Drr1[Drr1>=0.8]
        # Dtheta_new = Dtheta[Drr1>=0.8]
        # print('original poles:', Drr1.shape[0], 'new poles:', Drr_new.shape[0])
        # N = Drr_new.shape[0]*2

        # N = keep_index.shape[0] - 1
        net = classificationWSparseCode(num_class=args.num_class,
                                        Npole=args.N+1,
                                        Drr=Drr, Dtheta=Dtheta,
                                        dataType=args.dataType, dim=2,
                                        fistaLam=args.lam_f,
                                        gpu_id=args.gpu_id,
                                        useCL=False).cuda(args.gpu_id)
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                                    lr=args.lr, weight_decay=0.001, momentum=0.9)
        
        # net = load_pretrainedModel(stateDict, net)
        # net = load_pretrainedModel_endtoEnd(stateDict, net)
        # NOTE: Fixed DYAN poles'
        # ipdb.set_trace()
        net.sparseCoding.rr.requires_grad = False
        net.sparseCoding.theta.requires_grad = False
    elif args.mode == 'rgb':
        
        # kinetics_pretrain = './pretrained/i3d_kinetics.pth'
        pretrain = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/rhDYAN_bi_rgb_Att_2/100.pth'
        print('pretrain:', pretrain)
        stateDict = torch.load(pretrain, map_location=args.map_loc)['state_dict']
        net = classificationWBinarizationRGB( args.num_class, T=36,
                                             Npole=args.N+1, Drr=Drr, Dtheta=Dtheta,
                                             dataType='rgb', gpu_id=args.gpu_id).cuda(args.gpu_id)
        net.load_state_dict(stateDict)
        'fixed pretrained binarycoding, getRGB'
        # for params in net.binaryCoding.parameters():
        #     params.requires_grad = False
        # for params in net.getRGBFeature.parameters():
        #     params.requires_grad = False
        # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001,
        #                             momentum=0.9)
        optimizer = torch.optim.SGD([
            {'params': filter(lambda x: x.requires_grad, net.binaryCoding.parameters()), 'lr': 1e-4},
            {'params': filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr': args.lr},
            {'params': filter(lambda x: x.requires_grad, net.getRGBFeature.parameters()), 'lr': args.lr}],
            weight_decay=1e-4, momentum=0.9)
        # ipdb.set_trace()
    elif args.mode == 'rgb+dy':
        pretrain2 = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/rhDYAN_bi_rgb_Att_2/100.pth'
        pretrain1 = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/rhDYAN_bi/100.pth'
        net = classificationWBinarizationRGBDY(pretrain1, pretrain2, args.num_class, T=args.T,
                                               Npole=args.N+1,
                                               gpu_id=args.gpu_id).cuda(args.gpu_id)
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001,
                                    momentum=0.9)
    else: # with CL
        'NOTE: useCL --> if true, network is under CL setting, while false means no CL involved'
        net = contrastiveNet(dim_embed=128, Npole=args.N + 1, Drr=Drr, Dtheta=Dtheta,
                             Inference=True, gpu_id=args.gpu_id, dim=2,
                            dataType='2D', fistaLam=0.1,
                            fineTune=True, useCL=False).cuda(args.gpu_id)
        optimizer = torch.optim.SGD(
            [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()), 'lr': args.lr_2},
            {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()), 'lr': args.lr}],
            weight_decay=1e-4, momentum=0.9)
        stateDict = torch.load(args.dy_pretrain, map_location=args.map_loc)['state_dict']
        net = load_pretrainedModel_endtoEnd(stateDict, net)
        # print('check')

    # 
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    Criterion = torch.nn.CrossEntropyLoss()
    mseLoss = torch.nn.MSELoss()
    L1loss = torch.nn.SmoothL1Loss()
    cosSIM = nn.CosineSimilarity(dim=1, eps=1e-6)

    LOSS = []
    ACC = []
    LOSS_CLS = []
    LOSS_MSE = []
    LOSS_BI = []
    print('Experiment config(setup, clip, lam1, lam2, lr, lr_2, groupLasso):',
          args.setup, args.sampling, 
          args.Alpha, args.lam1, args.lam2,
          args.lr, args.lr_2, args.groupLasso)
    for epoch in range(1, args.Epoch+1):
        print('start training epoch:', epoch)
        lossVal = []
        lossCls = []
        lossBi = []
        lossMSE = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            
            # print('sample:', i)
            optimizer.zero_grad()

            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            images = sample['input_images'].float().cuda(args.gpu_id)
            ROIs = sample['input_rois'].float().cuda(args.gpu_id)
            visibility = sample['input_skeletons']['visibility'].float().cuda(args.gpu_id)
            gt_label = sample['action'].cuda(args.gpu_id)

            if args.sampling == 'Single':
                t = skeletons.shape[1]
                input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)  #bz, T, 25, 2
                input_mask = visibility.reshape(visibility.shape[0], t, -1)
                nClip = 1
                input_images = images
                input_rois = ROIs

            else:
                t = skeletons.shape[2]
                input_skeletons = skeletons.reshape(skeletons.shape[0], skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
                input_mask = visibility.reshape(visibility.shape[0]*visibility.shape[1], t, -1)
                input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
                input_rois = ROIs.reshape(ROIs.shape[0]*ROIs.shape[1], t, 3, 224, 224)
                nClip = skeletons.shape[1]

            if args.withMask:
                input_skeletons = input_skeletons.unsqueeze(-1)
                input_mask = input_mask.unsqueeze(-1) + torch.tensor(0.01).float().cuda(args.gpu_id)
                # input_mask = torch.ones_like(input_skeletons).float().cuda(gpu_id)
            else:
                input_mask = torch.ones(1).cuda(args.gpu_id)
                # gt_skeletons = input_skeletons

            if args.mode == 'dy+cl':
                # NOTE: dy+cl
                keep_index = None
                actPred, output_skeletons, lastFeat = net(input_skeletons, t) #bi_thresh=gumbel threshold
                # actPred, output_skeletons,_ = net.forward2(input_skeletons, t, keep_index)
                actPred = actPred.reshape(skeletons.shape[0], nClip, args.num_class)
                actPred = torch.mean(actPred, 1)

                # loss = lam1 * Criterion(actPred, gt_label) + lam2 * mseLoss(output_skeletons, input_skeletons)
                loss = Criterion(actPred, gt_label) #'with fixed dyan'
                lossMSE.append(mseLoss(output_skeletons, input_skeletons).data.item())
                lossBi.append(0)
            elif args.mode == 'rgb':
                keep_index = None
                b, t, c, x, y = input_images.shape
                # ipdb.set_trace()
                actPred, binaryCode, output_feat,input_feat = net(input_images.reshape(b*t,c, x, y), input_rois.reshape(b*t, c, x, y))
                # actPred, _ = net(input_images, input_rois)
                # ipdb.set_trace()
                bi_gt = torch.zeros_like(binaryCode).cuda(args.gpu_id)
                actPred = actPred.reshape(skeletons.shape[0], nClip, args.num_class)
                actPred = torch.mean(actPred, 1)    
                
                # loss = Criterion(actPred, gt_label)
                # print(gt_label)
            
                loss = args.lam1 * Criterion(actPred, gt_label) + args.lam2 * mseLoss(output_feat, input_feat) \
                    + args.Alpha * L1loss(binaryCode, bi_gt)
                # loss = mseLoss(output_feat, input_feat) + 0.1 * L1loss(binaryCode, bi_gt)
                # lossMSE.append(mseLoss(output_skeletons, input_skeletons.squeeze(-1)).data.item())
                # print('output shape:', output_skeletons.shape, 'mse:', mseLoss(output_skeletons, input_skeletons).data.item())

            
                lossMSE.append(mseLoss(output_feat, input_feat).data.item())
                lossBi.append(L1loss(binaryCode, bi_gt).data.item())

            elif args.mode == 'rgb+dy':
                keep_index = None
                b, t, c, x, y = input_images.shape
                actPred  = net(input_skeletons, t,input_images.reshape(b*t,c, x, y), input_rois.reshape(b*t, c, x, y))
                loss = Criterion(actPred, gt_label)
            
            else:
                input_seletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
                actPred, lastFeat, binaryCode, output_skeletons = net(input_skeletons, args.gumbel_thresh)
                bi_gt = torch.zeros_like(binaryCode).cuda(args.gpu_id)
                actPred = actPred.reshape(skeletons.shape[0], nClip, args.num_class)
                actPred = torch.mean(actPred, 1)    

                loss = args.lam1 * Criterion(actPred, gt_label) + args.lam2 * mseLoss(output_skeletons, input_skeletons.squeeze(-1)) \
                    + args.Alpha * L1loss(binaryCode, bi_gt)

                lossMSE.append(mseLoss(output_skeletons, input_skeletons.squeeze(-1)).data.item())
            
                lossBi.append(L1loss(binaryCode, bi_gt).data.item())


            loss.backward()
            # ipdb.set_trace()
            optimizer.step()
            lossVal.append(loss.data.item())

            lossCls.append(Criterion(actPred, gt_label).data.item())


        loss_val = np.mean(np.array(lossVal))
        # LOSS.append(loss_val)
        # LOSS_CLS.append(np.mean(np.array((lossCls))))
        # LOSS_MSE.append(np.mean(np.array(lossMSE)))
        # LOSS_BI.append(np.mean(np.array(lossBi)))
        # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)), '|bi:', np.mean(np.array(lossBi)))
        print('epoch:', epoch, 'loss:', loss_val)
        end_time = time.time()
        print('training time(h):', (end_time - start_time)/3600)

        scheduler.step()
        
        if epoch % 1 == 0:

            Acc = testing(testloader, net, args.gpu_id, args.sampling,
                          args.mode, args.withMask,
                          args.gumbel_thresh,
                          keep_index)
            print('testing epoch:',epoch, 'Acc:%.4f'% Acc)
            ACC.append(Acc)

        
    torch.cuda.empty_cache()
    print('done')

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)
    if args.sampling != 'Single':
        args.bs = 4
        args.nw = 4
    if args.RHdyan:
        if args.maskType == 'binary':
            args.dy_pretrain = './pretrained/N-UCLA/' + args.setup + '/' + args.sampling + '/pretrainedRHdyan_BI_mask.pth'  # binary mask
        elif args.maskType == 'score':
            args.dy_pretrain = './pretrained/N-UCLA/' + args.setup + '/' + args.sampling + '/pretrainedRHdyan_BI_score.pth'
        else:
            # dy_pretrain = './pretrained/NUCLA/' + setup + '/' + sampling +'/pretrainedRHdyan_noCL.pth'
           args.dy_pretrain = './pretrained/NUCLA/' + args.setup + '/' + args.sampling + '/pretrainedRHdyan_noCL_v2.pth'
    else:
        args.dy_pretrain = './pretrained/N-UCLA/' + args.setup + '/' + args.sampling + '/pretrainedDyan_BI.pth'
    main(args)