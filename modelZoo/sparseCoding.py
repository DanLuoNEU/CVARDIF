############################# Import Section #################################
import sys
sys.path.append('../')
sys.path.append('../data')
sys.path.append('.')

import ipdb
## Imports related to PyTorch
import time
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
# from modelZoo.actHeat import imageFeatureExtractor
import torch
from math import sqrt
import numpy as np
import pdb
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
############################# Import Section #################################


def creatRealDictionary(T, rr, theta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones  = Wones
    for i in range(0,T):
        W1 = torch.mul(torch.pow(rr,i) , torch.cos(i * theta))
        W2 = torch.mul (torch.pow(rr,i) , torch.sin(i *theta) )
        W = torch.cat((Wones,W1,W2),0)
        WVar.append(W.view(1,-1))
    dic = torch.cat((WVar),0)

    # G = torch.norm(dic,p=2,dim=0)
    # # idx = (G == 0).nonzero()
    # idx = G==0
    # nG = G.clone()
    # # print(type(T))
    # nG[idx] = np.sqrt(T)
    # G = nG
    #
    # dic = dic/G

    return dic

def fista_new(D, Y, lambd,maxIter, gpu_id):
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD,2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id)
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,linv)
    DtY = torch.mul(DtY,linv)
    Softshrink = nn.Softshrink(lambd)
    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Softshrink((Ay + DtY)+1e-6)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-5:
            x_old = x_new
            # print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old

def fista_reweighted(D, Y, lambd, w, maxIter):
    'D: T x 161, Y: N x T x 50, w: N x 161 x 50'
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0, 2, 1), D)
        DtY = torch.matmul(D.permute(0, 2, 1), Y)
    # eig, v = torch.eig(DtD, eigenvectors=True)
    # eig, v = torch.linalg.eig(DtD)
    # L = torch.max(eig)
    L = torch.norm(DtD, 2)

    # eigs = torch.abs(torch.linalg.eigvals(DtD))
    # L = torch.max(eigs, dim=1).values

    Linv = 1/L


    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).to(D.device) - torch.mul(DtD,Linv)
    t_old = 1

    const_xminus = torch.mul(DtY, Linv) - weightedLambd.to(D.device)
    const_xplus = torch.mul(DtY, Linv) + weightedLambd.to(D.device)

    iter = 0

    while iter < maxIter:
        iter +=1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old-1)/t_new
        y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old

class DyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, lam, gpu_id):
        super(DyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T
        self.lam = lam
        self.gpu_id = gpu_id

    def forward(self, x,T):
        '''with Reweighted Heuristic Algorithm
        '''
        dic = creatRealDictionary(T, self.rr,self.theta, self.gpu_id)
        # print('rr:', self.rr, 'theta:', self.theta)
        # sparseCode = fista_new(dic,x,self.lam, 100,self.gpu_id)

        i = 0
        # ipdb.set_trace()
        w_init = torch.ones(x.shape[0], dic.shape[1], x.shape[2])
        while i < 2:
            temp = fista_reweighted(dic, x, self.lam, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-2)
            w_init = (w/torch.norm(w)) * dic.shape[1]

            'for matrix:'
            # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
            # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
            # pdb.set_trace()

            final = temp
            del temp
            i += 1

        sparseCode = final
        reconst = torch.matmul(dic, sparseCode.cuda(self.gpu_id))
        return sparseCode, dic, reconst
    
    def forward2(self,x, T):
        """Without Reweighted Heuristic Algorithm
        """
        dic = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        sparseCode = fista_new(dic, x, self.lam, 100, self.gpu_id)
        
        reconst = torch.matmul(dic, sparseCode)
        # return sparseCode, dic, reconst
        return sparseCode, dic, reconst
    
    def prediction(self,x, FRA, PRE):
        # c,_,_ = self.forward2(x,FRA) # regular dyan
        c, _, _ = self.forward(x, FRA) # rh dyan
        dict_new = creatRealDictionary(PRE+FRA, self.rr, self.theta, self.gpu_id)
        pred = torch.matmul(dict_new, c)
        return pred, c, dict_new
    
    def thresholdDict(self, x, T, keep_index):
        dict = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        # ipdb.set_trace()
        dict_new = dict[:,keep_index]

        i = 0
        w_init = torch.ones(x.shape[0], dict_new.shape[1], x.shape[2])
        while i < 2:
            temp = fista_reweighted(dict_new, x, self.lam, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-2)
            w_init = (w/torch.norm(w)) * dict_new.shape[1]

            'for matrix:'
            # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
            # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
            # pdb.set_trace()

            final = temp
            del temp
            i += 1

        sparseCode = final
        reconst = torch.matmul(dict_new, sparseCode.cuda(self.gpu_id))
        return sparseCode, dict_new, reconst

    
def fista_reweighted_mask(D, Y, lambd, w, maxIter):
    'D: T x 50, Y: bz x T x 1, w:bz x 161 x 1 '
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0,2,1), D)
        DtY = torch.matmul(D.permute(0,2,1), Y)

    eigs = torch.abs(torch.linalg.eigvals(DtD))
    L = torch.max(eigs, dim=1).values


    # print('max eigen:', L, 'min eigen:', torch.min(eigs), 'max D:', torch.max(D))
    # pdb.set_trace()
    # L = torch.norm(DtD, 2)
    Linv = (1 / L).unsqueeze(1).unsqueeze(2)
    # print(w)
    # pdb.set_trace()
    # w = w.permute(2, 1, 0)
    weightedLambd = (w * lambd) * Linv

    x_old = torch.zeros(DtD.shape[0], DtY.shape[1], DtY.shape[2]).to(D.device)

    y_old = x_old
    I = torch.eye(DtD.shape[1]).to(D.device).unsqueeze(0).repeat(DtD.shape[0], 1, 1)
    A = I - DtD * (Linv)
    t_old = 1

    const_xminus = DtY * Linv - weightedLambd
    const_xplus = DtY * Linv + weightedLambd

    iter = 0

    while iter < maxIter:

        iter += 1
        Ay = torch.bmm(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old - 1) / t_new
        # print(t_new)
        y_new = x_new + torch.mul(tt, (x_new - x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

        del t_new, x_new, y_new
        # torch.cuda.empty_cache()
    return x_old


def fista_reweighted_batch(D, Y, lambd, w, maxIter, gpu_id):
    # Y = Y.squeeze(0)
    # Y = Y.permute(1, 0)
    # Y = Y.unsqueeze(2)
    if len(Y.shape) == 3:
        Y = Y.permute(2, 1, 0)
        Dt = D.permute(0, 2, 1)
        w = w.permute(2, 1, 0)
    else:
        Y = Y.permute(0, 2, 1, 3)
        Dt = D.permute(0, 1, 3, 2)
        w = w.permute(0, 3, 2, 1)

    DtD = torch.matmul(Dt, D)
    DtY = torch.matmul(Dt, Y)

    eigs = torch.abs(torch.linalg.eigvals(DtD))
    L = torch.max(eigs, dim=-1).values

    # print('max eigen:', L, 'min eigen:', torch.min(eigs), 'max D:', torch.max(D))
    # pdb.set_trace()
    Linv = 1 / L
    # print(w)
    # pdb.set_trace()

    weightedLambd = (w * lambd) * Linv.unsqueeze(-1).unsqueeze(-1)

    if len(Y.shape) == 3:
        x_old = torch.zeros(DtD.shape[0], DtY.shape[1], DtY.shape[2]).to(D.device)

        I = torch.eye(DtD.shape[1]).to(D.device).unsqueeze(0).repeat(DtD.shape[0], 1, 1)

    else:
        x_old = torch.zeros(DtD.shape[0], DtD.shape[1], DtY.shape[2], DtY.shape[3]).to(D.device)
        I = torch.eye(DtD.shape[2]).to(D.device).unsqueeze(0).unsqueeze(0).repeat(DtD.shape[0],DtD.shape[1],1,1)

    y_old = x_old


    A = I - DtD * (Linv.unsqueeze(-1).unsqueeze(-1))

    t_old = 1

    const_xminus = DtY * (Linv.unsqueeze(-1).unsqueeze(-1)) - weightedLambd
    const_xplus = DtY * (Linv.unsqueeze(-1).unsqueeze(-1)) + weightedLambd

    iter = 0

    while iter < maxIter:
        iter += 1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old - 1) / t_new
        y_new = x_new + torch.mul(tt, (x_new - x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old


'''
This function is the implementation of the reweighted fista with weight of coefficient as an input.
'''


def fista_reweighted_wc_batch(D, Y, lambd, Wc, maxIter, gpu_id):
    i = 0
    w_init = Wc
    # w_init = torch.ones(1, dic.shape[1], x.shape[2]).cuda(self.gpu_id)
    while i < 2:
        temp = fista_reweighted_batch(D, Y, lambd, w_init, maxIter, gpu_id)
        'for vector:'
        w = Wc / (torch.abs(temp) + 1e-2)
        w_init = (w / torch.norm(w)) * D.shape[1]

        'for matrix:'
        # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
        # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
        # pdb.set_trace()
        final = temp
        del temp
        i += 1
    sparseCode = final

    reconst = torch.matmul(D, sparseCode)
    return sparseCode, D, reconst


def fista_reweighted_mask_batch(D, Y, lambd, w, M, maxIter, gpu_id):
    YM = M * Y
    if len(Y.shape) == 3:
        DM = D * M.permute(2, 1, 0)
    else:
        DM = D * M.permute(0, 2, 1, 3)
    return fista_reweighted_batch(DM, YM, lambd, w, maxIter, gpu_id)


def fista_reweighted_mask_wc_batch(D, Y, lambd, Wc, M, maxIter, gpu_id):
    i = 0
    w_init = Wc
    # w_init = torch.ones(1, dic.shape[1], x.shape[2]).cuda(self.gpu_id)
    while i < 2:
        temp = fista_reweighted_mask_batch(D, Y, lambd, w_init, M, maxIter, gpu_id)
        'for vector:'
        if len(Y.shape) == 3:
            w = Wc / ((torch.abs(temp) + 1e-2).permute(2, 1, 0))
            w_init = (w / torch.norm(w)) * D.shape[-1]
        else:
            w = Wc/((torch.abs(temp) + 1e-2).permute(0,3,2,1))
            w_init = (w/torch.norm(w)) * D.shape[-1] * D.shape[-2]

        'for matrix:'
        # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
        # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
        # pdb.set_trace()
        final = temp
        del temp
        i += 1
    sparseCode = final

    reconst = torch.matmul(D, sparseCode)
    return sparseCode, D, reconst


class MaskDyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, lam, gpu_id):
        super(MaskDyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T
        self.lam = lam
        self.gpu_id = gpu_id

    def forward(self, x, T, M):
        dic = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        if len(x.shape) == 3:
            dic = dic.repeat(x.shape[2], 1, 1)
            Wc = torch.ones(1, dic.shape[2], x.shape[-2]).cuda(self.gpu_id)
        else:
            dic = dic.repeat(x.shape[0], x.shape[2], 1, 1)
            Wc = torch.ones(x.shape[0], x.shape[-1], dic.shape[-1], x.shape[-2]).cuda(self.gpu_id)
        # print('rr:', self.rr, 'theta:', self.theta)


        sparseCode, dic, reconst = fista_reweighted_mask_wc_batch(dic, x, self.lam, Wc, M, 100, self.gpu_id)
        if len(x.shape) == 3:
            sparseCode = sparseCode.permute(2,1,0)
            dic = torch.mean(dic,0)
        else:
            sparseCode = sparseCode.squeeze(-1).permute(0,2,1)
            dic = torch.mean(torch.mean(dic,0),0)


        return sparseCode, dic, reconst


if __name__ == '__main__':
    modelRoot = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/'
    saveModel = modelRoot + 'Single/regularDYAN_seed123/'
    if not os.path.exists(saveModel):
        os.makedirs(saveModel)
    lam = 0.1
    N = 80 * 2
    Epoch = 100
    gpu_id = 2
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    withMask = False
    net = DyanEncoder(Drr, Dtheta, lam, gpu_id)
    # net = binarizeSparseCode(num_binary=128, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id, Inference=True, fistaLam=lam)
    net.cuda(gpu_id)
    setup = 'setup1'  # v1,v2 train, v3 test;
    path_list = '../data/CV/' + setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'

    trainSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling='Single', phase='train', T=36,
                               maskType='score',
                               setup=setup)
    # #

    trainloader = DataLoader(trainSet, batch_size=12, shuffle=True, num_workers=4)

    testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling='Single', phase='test', T=36,
                              maskType='score', setup=setup)
    testloader = DataLoader(testSet, batch_size=12, shuffle=True, num_workers=4)

    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001,
                                momentum=0.9)

    net.train()

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 70], gamma=0.1)

    mseLoss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss(reduction='sum')
    # Loss = []
    for epoch in range(0, Epoch+1):
        print('training epoch:', epoch)
        lossVal = []
        lossMSE = []
        lossL1 = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()

            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

            t = skeletons.shape[1]
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

            'regular dyan:'
            sparseCode,_,output_skeletons = net.forward2(input_skeletons, t)
            # output_skeletons = net.prediction(input_skeletons[:,0:t-1], t-1)
            # loss = 0.02*mseLoss(output_skeletons[:,0:t-1], input_skeletons[:,0:t-1]) + 1*mseLoss(output_skeletons[:,-1], input_skeletons[:,-1])
            loss = mseLoss(output_skeletons, input_skeletons)
            # + 1e-4*l1Loss(sparseCode,target_coeff)
            # ipdb.set_trace()
            
            'rh-dyan + bi'
            
            # binaryCode, reconstruction = net(input_skeletons, t)
            # target_coeff = torch.zeros_like(binaryCode).cuda(gpu_id)
            # loss = mseLoss(output_skeletons, input_skeletons) + 0.1*l1Loss(binaryCode,target_coeff)
            loss.backward()
            # print('rr.grad:', net.rr.grad[0:10], 'theta.grad:', net.theta.grad[0:10])
            optimizer.step()
            # ipdb.set_trace()
            lossMSE.append(mseLoss(output_skeletons, input_skeletons).data.item())
            # lossL1.append(0.1*l1Loss(binaryCode,target_coeff).data.item())
            lossVal.append(loss.data.item())
        end_time = time.time()
        print('epoch:', epoch, 'loss:', np.mean(np.asarray(lossVal)), 'time(h):', (end_time - start_time) / 3600)
        # print('epoch:', epoch, 'mse loss:', np.mean(np.asarray(lossMSE)), 'L1 loss:', np.mean(np.asarray(lossL1)))

        if epoch % 5 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')
            with torch.no_grad():
                ERROR = torch.zeros(testSet.__len__(), 1)

                for i, sample in enumerate(testloader):
                    
                    skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

                    t = skeletons.shape[1]
                    input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

                    _,_,output_skeletons = net.forward2(input_skeletons, t) # reconst
                    # output_skeletons = net.prediction(input_skeletons[:,0:t-1], t-1)
                    # _, output_skeletons = net(input_skeletons, t)
                    # ipdb.set_trace()
                    # error = torch.norm(output_skeletons[:,-1] - input_skeletons[:,-1]).cpu()
                    error = torch.norm(output_skeletons - input_skeletons).cpu()
                    ERROR[i] = error

                print('epoch:', epoch, 'error:', torch.mean(ERROR))

        scheduler.step()
    print('done')