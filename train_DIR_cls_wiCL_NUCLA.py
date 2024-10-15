# DIR-3: train CVAR cls with CL feature, N-UCLA dataset
import time
import argparse
from matplotlib import pyplot as plt

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.crossView_UCLA import NUCLA_CrossView,random,np,os
from modelZoo.BinaryCoding import nn, gridRing, contrastiveNet
from utils import load_pretrainedModel
from test_cls_CV import testing, getPlots

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='CVARDIF')
    parser.add_argument('--modelRoot', default='/data/Dan/202111_CVAR/NUCLA/DIR_cls_wiCL',
                        help='the work folder for storing experiment results')
    parser.add_argument('--cus_n', default='', help='customized name')
    
    parser.add_argument('--dataset', default='NUCLA', help='') # dataset = 'NUCLA'
    parser.add_argument('--setup', default='setup1', help='') # setup = 'setup1' # v1,v2 train, v3 test;
    parser.add_argument('--num_class', default=10, type=int, help='') # num_class = 10
    parser.add_argument('--dataType', default='2D', help='') # dataType = '2D'
    parser.add_argument('--sampling', default='Single', help='') # sampling = 'Single' #sampling strategy
    parser.add_argument('--bs', default=4, type=int, help='')
    parser.add_argument('--nw', default=12, type=int, help='')
    
    parser.add_argument('--mode', default='dy+bi+cl', help='dy+bi+cl | dy+cl | rgb+dy') # mode = 'dy+bi+cl'
    parser.add_argument('--RHdyan', default='1', type=str2bool, help='') # RHdyan = True
    parser.add_argument('--withMask', default='0', type=str2bool, help='') # withMask = False
    parser.add_argument('--maskType', default='None', help='') # maskType = 'score'
    parser.add_argument('--contrastive', default='1', type=str2bool, help='') # constrastive = True
    parser.add_argument('--fusion', default='0', type=str2bool, help='') # fusion = False
    parser.add_argument('--groupLasso', default='0', type=str2bool, help='')


    parser.add_argument('--T', default=36, type=int, help='') # T = 36 # input clip length
    parser.add_argument('--N', default=80*2, type=int, help='') # N = 80*2
    parser.add_argument('--lam_f', default=0.1, type=float) # fistaLam = 0.1
    parser.add_argument('--gumbel_thresh', default=0.5, type=float) # 0.503 # gumbel_thresh = 0.505

    parser.add_argument('--gpu_id', default=7, type=int, help='') # gpu_id = 7
    parser.add_argument('--Epoch', default=100, type=int, help='') # Epoch = 100
    parser.add_argument('--lr', default=1e-3, type=float, help='classifier') # lr = 1e-3 # classifier
    parser.add_argument('--lr_2', default=1e-3, type=float, help='sparse coding') # lr_2 = 1e-3  # sparse codeing
    parser.add_argument('--Alpha', default=1e-2, type=float, help='bi loss')
    parser.add_argument('--lam1', default=2, type=float, help='cls loss')
    parser.add_argument('--lam2', default=0.5, type=float, help='mse loss')

    return parser

def main(args):
    '------configuration:-------------------------------------------'
    args.saveModel = args.modelRoot + f"{args.sampling}_{args.mode}_T36_wiCL/"
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    print('mode:',args.mode, 'model path:', args.saveModel, 'mask:', args.maskType)
    '============================================= Main Body of script================================================='
    P,Pall = gridRing(args.N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    path_list = f'/home/dan/ws/202209_CrossView/202409_CVAR_yuexi_lambdaS/data/CV/{args.setup}/' # path_list = './data/CV/' + setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               setup=args.setup, dataType=args.dataType,
                               sampling=args.sampling, nClip=args.nClip,
                               T=args.T, maskType=args.maskType)
    trainloader = DataLoader(trainSet, shuffle=True,
                             batch_size=args.bs, num_workers=args.nw)
    # testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, sampling=sampling, phase='test', cam='2,1', T=T, maskType= maskType, setup=setup)
    # testloader = DataLoader(testSet, batch_size=bz, shuffle=True, num_workers=num_workers)

    net = contrastiveNet(dim_embed=128, Npole=args.N+1,
                         Drr=Drr, Dtheta=Dtheta, fistaLam=args.lam_f,
                         Inference=True, nClip=args.nClip,
                         dim=2, mode='rgb',
                         fineTune=False, useCL=True,
                         gpu_id=args.gpu_id).cuda(args.gpu_id)
    net.train()

    if args.mode != 'rgb':
        pre_trained = './pretrained/NUCLA/setup1/Multi/pretrainedRHdyan_for_CL.pth'
        state_dict = torch.load(pre_trained, map_location=args.map_loc)['state_dict']
        net = load_pretrainedModel(state_dict, net)

        optimizer = torch.optim.SGD(
                [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()),
                  'lr': args.lr_2},
                {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()),
                 'lr': args.lr}], weight_decay=1e-3, momentum=0.9)

    optimizer = torch.optim.SGD(
                [{'params': filter(lambda x: x.requires_grad, net.parameters()),
                  'lr': args.lr_2}], weight_decay=1e-3, momentum=0.9)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    Criterion = torch.nn.CrossEntropyLoss()
    mseLoss = torch.nn.MSELoss()
    L1loss = torch.nn.SmoothL1Loss()
    cosSIM = nn.CosineSimilarity(dim=1, eps=1e-6)

    LOSS = []
    ACC = []

    LOSS_CLS = []
    LOSS_MSE = []
    LOSS_BI = []
    print('experiment setup:',args.RHdyan, args.constrastive)
    for epoch in range(1, args.Epoch+1):
        print('start training epoch:', epoch)
        lossVal = []

        start_time = time.time()
        for i, sample in enumerate(trainloader):

            # print('sample:', i)
            optimizer.zero_grad()

            # skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
            # skeletons = sample['input_skeletons']['unNormSkeleton'].float().cuda(gpu_id)
            skeletons = sample['input_skeletons']['affineSkeletons'].float().cuda(args.gpu_id)
            visibility = sample['input_skeletons']['visibility'].float().cuda(args.gpu_id)
            gt_label = sample['action'].cuda(args.gpu_id)
            # ipdb.set_trace()
            if args.sampling == 'Single':
                t = skeletons.shape[2]
                input_skeletons = skeletons.reshape(skeletons.shape[0],skeletons.shape[1], t, -1)  #bz, 2, T, 25, 2
                input_mask = visibility.reshape(visibility.shape[0], t, -1)
                nClip = 1

            else:
                t = skeletons.shape[3]
                input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], skeletons.shape[2], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
                input_mask = visibility.reshape(visibility.shape[0]*visibility.shape[1], t, -1)
                nClip = skeletons.shape[1]

            # info_nce_loss = net(input_skeletons, t)
            x = input_skeletons
            y = 0.501 # bi_threshold
            logits, labels = net(x, y)
            info_nce_loss = Criterion(logits, labels)
            info_nce_loss.backward()
            # ipdb.set_trace()
            optimizer.step()
            lossVal.append(info_nce_loss.data.item())
        scheduler.step()
        print('epoch:', epoch, 'contrastive loss:', np.mean(np.asarray(lossVal)))
        # print('rr.grad:', net.backbone.sparseCoding.rr.grad, 'cls grad:', net.backbone.Classifier.cls[-1].weight.grad[0:10,0:10])
        if epoch % 10 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.saveModel + str(epoch) + '.pth')



if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id) # map_loc = "cuda:"+str(gpu_id)
    if args.sampling == 'Single':
        args.nClip = 1
        args.bs = 12
        args.nw = 8
    else:
        args.nClip = 4
        args.bs = 20
        args.nw = 8
    # if args.RHdyan:
    #     if args.maskType == 'binary':
    #         args.dy_pretrain = './pretrained/N-UCLA/' + args.setup + '/' + args.sampling + '/pretrainedRHdyan_BI_mask.pth'  # binary mask
    #     elif args.maskType == 'score':
    #         args.dy_pretrain = './pretrained/N-UCLA/' + args.setup + '/' + args.sampling + '/pretrainedRHdyan_BI_score.pth'
    #     else:
    #         # dy_pretrain = './pretrained/NUCLA/' + setup + '/' + sampling +'/pretrainedRHdyan_noCL.pth'
    #        args.dy_pretrain = './pretrained/NUCLA/' + args.setup + '/' + args.sampling + '/pretrainedRHdyan_noCL_v2.pth'
    # else:
    #     args.dy_pretrain = './pretrained/N-UCLA/' + args.setup + '/' + args.sampling + '/pretrainedDyan_BI.pth'
    main(args)
    # 'plotting results:'
    # getPlots(LOSS,LOSS_CLS, LOSS_MSE, LOSS_BI, ACC,fig_name='DY_CL.pdf')
    torch.cuda.empty_cache()
    print('done')