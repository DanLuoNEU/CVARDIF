############################# Import Section #################################
import sys
from math import sqrt
import numpy as np

import torch
import torch.nn as nn
# sys.path.append('../')
# sys.path.append('../data')
# sys.path.append('.')
from utils import ipdb, random
############################# Import Section #################################
def creatRealDictionary(T, rr, theta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones  = Wones
    for i in range(0,T):
        W1 = torch.mul(torch.pow(rr,i) , torch.cos(i * theta))
        W2 = torch.mul(torch.pow(rr,i) , torch.sin(i * theta) )
        W = torch.cat((Wones,W1,W2),0)
        WVar.append(W.view(1,-1))
    dic = torch.cat((WVar),0)

    return dic


def fista_new(D, Y, lambd, maxIter, gpu_id):
    """ Original FISTA for LASSO. \\
            arg min_C = ||Y-DC||^2_2 + λ|C|^1_1 \\
        D: (T, num_pole)
        Y: ((batch_size x num_clips), T, num_joints x dim_joints)
    """
    DtD = torch.matmul(torch.t(D),D)
    # L = torch.norm(DtD,2)
    # L = spectral_norm_svd(DtD)
    L = torch.linalg.norm(DtD, ord=2)
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
        # if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-5:
        if torch.linalg.norm((x_old - x_new))/x_old.shape[1] < 1e-5:
            x_old = x_new
            # print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    
    return x_old

def fista_efc(D, Y, lambd, maxIter):
    """ Original FISTA for LASSO: \
            argmin_C = ||Y-DC||^2_2 + λ|C|^1_1 \
    INPUTS:
        D(torch.Tensor): Dictionary from DYAN (T, num_pole)
        Y(torch.Tensor): Input Data ((batch_size x num_clips), T, num_joints x dim_joints)
        lam(float): Regularization parameter
        max_iters (int): Maximum number of iterations.
    """
    # Move to GPU if available
    device = D.device
    # Variable repeatedly used
    DtD = D.T @ D
    # Compute Lipschitz constant and step size
    L_inv = 1/torch.linalg.norm(DtD, ord=2)

    num_pole, dim_data = DtD.shape[1],Y.shape[-1]
    A = torch.eye(num_pole, device=device) - L_inv * DtD
    B = L_inv * (D.T @ Y)
    Softshrink = nn.Softshrink(lambd*L_inv.item())
    
    x = torch.zeros(num_pole, dim_data, device=device)
    y = torch.zeros(num_pole, dim_data, device=device)
    t = 1

    del DtD
    for _ in range(maxIter):
        # Gradient step
        x_new = Softshrink( A @ y + B)
        # Check convergence
        if torch.linalg.norm((x - x_new))/dim_data < 1e-5:
            del A, B

            return x_new
        # Momentum update
        t_new = (1. + np.sqrt(1 + 4 * t**2)) / 2.
        tt = ( t - 1 ) / t_new
        y_new = (1 + tt) * x_new - tt * x
        # Update variables
        x = x_new
        t = t_new
        y = y_new
    del A, B

    return x


def fista_reweighted(D, Y, lambd, w, maxIter):
    '''
        D: [T, 161]\\
        Y: [Nsample, T, 50]\\
        w: [Nsample, 161, 25 x 2]
    '''
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0, 2, 1), D)
        DtY = torch.matmul(D.permute(0, 2, 1), Y)
    # Spectral norm/largest singular value of DtD
    # # Option 1: PyTorch < 1.9.0
    # L = spectral_norm_svd(DtD)
    # Option 2: PyTorch >=1.9.0
    L = torch.linalg.norm(DtD, ord=2)

    # # Here uses a highly optimized algorithm to estimate the largest singular value instead
    # # Ensure DtD is not rank-deficient or ill-conditioned, otherwise, this can lead to differences in results
    # L = torch.norm(DtD, 2) # element-wise norm
    Linv = 1/L
    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).to(D.device) - torch.mul(DtD,Linv)
    const_xminus = torch.mul(DtY, Linv) - weightedLambd.to(D.device)
    const_xplus = torch.mul(DtY, Linv) + weightedLambd.to(D.device)

    t_old = 1
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
        # if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
        if torch.linalg.norm((x_old - x_new))/x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old


class DyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, 
                 wiRH,
                 lam,
                 gpu_id, freezeD=False):
        super(DyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T
        self.wiRH = wiRH
        self.lam = lam
        self.gpu_id = gpu_id
        self.freezeD = freezeD
        # Frozen Dictionary, no need to create each forward
        if freezeD:  self.creat_D()

    def creat_D(self):
        """Create the Dictionary if the rho and theta are fixed
        """
        D = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
        # self.D = D/torch.norm(D,dim=0, keepdim=True)
        # self.D = D/torch.linalg.norm(D, dim=0, keepdim=True)

    def FISTA(self, x, dic):
        """
        """
        # C = fista_new(dic, x, self.lam, 100, self.gpu_id) # Yuexi version
        C = fista_efc(dic, x, self.lam, 100) # Dan version
        # Reconstruction
        reconst = torch.matmul(dic, C)

        return C, dic, reconst

    def RH_FISTA(self, x, dic):
        '''With Reweighted Heuristic Algorithm
        x: N x T x (num_j x dim_j)
        '''
        # if self.freezeD: 
        #     D = self.D.detach() # Detach to ensure it's not part of a computation graph
        # else:            
        #     D = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
        #     ipdb.set_trace()
        #     D = D/torch.norm(D,dim=0, keepdim=True)
        #     ipdb.set_trace()
        # batch_size, num_poles, num_joints x dim_joint
        Nsample, Npole, Ddata = x.shape[0], dic.shape[1], x.shape[2]
        w_init = torch.ones(Nsample, Npole, Ddata)

        i = 0
        while i < 2:
            # temp: [N, Np, num_j x dim_j]
            temp = fista_reweighted(dic, x, self.lam, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-2)
            # ipdb.set_trace()
            # Scaler for Reweight
            # Batch-wise
            # w_init = (w/torch.norm(w)) * D.shape[1]
            # Matrix-wise
            # w_init = (w/torch.norm(w, dim=(1,2)).view(-1,1,1)) * Npole
            # Column-wise for Coefficients
            # w_init = (w/torch.norm(w, dim=(1)).view(Nsample,1,-1)) * Npole # norm 2
            # w_init = (w/torch.norm(w, p=1, dim=(1)).view(Nsample,1,-1)) * Npole # norm 1, pytorch<1.9.0
            w_init = (w/torch.linalg.norm(w, ord=1, dim=1, keepdim=True)) * Npole # norm 1, pytorch>=1.9.0

            
            final = temp
            del temp
            i += 1

        sparseCode = final
        reconst = torch.matmul(dic, sparseCode.cuda(self.gpu_id))

        return sparseCode, dic, reconst
    
    def tensor_hook1(self, grad):
        print(f"Gradient of rr: {grad}")
        ipdb.set_trace()

    def tensor_hook2(self, grad):
        print(f"Gradient of theta: {grad}")
        ipdb.set_trace()

    def forward(self, x):
        # print('rr: max', self.rr.max(),'min',self.rr.min(),'mean', self.rr.mean())
        # print('theta: max', self.theta.max(), 'min', self.theta.min(), 'mean',self.theta.mean())
        # self.rr.register_hook(self.tensor_hook1)
        # self.theta.register_hook(self.tensor_hook2)
        dic = self.D.detach() if self.freezeD else creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
        # dic = dic/torch.linalg.norm(dic, dim=0, keepdim=True)
        if self.wiRH:   return self.RH_FISTA(x,dic)
        else:           return self.FISTA(x,dic)


def real_D(T, rho, theta, 
           CONSTR=True,
           NORM=True):
    """ Create Real Dictionary
    INPUT:
        T:int, sequence length
        rho:torch.tensor(float), (N)
        theta:torch.tensor(float),(N)
    """
    if CONSTR:
        rmin, rmax = 0.001, 1.15
        tmin, tmax = 0., torch.pi
        rho = rmin + (rmax-rmin) * torch.sigmoid(rho)
        theta = tmin + (tmax-tmin) * torch.sigmoid(theta)
    Wones = torch.ones(1).to(rho)

    WVar = []
    for i in range(0,T):
        W1 = torch.pow(rho, i) * torch.cos(i * theta)
        W2 = torch.pow(rho, i) * torch.sin(i * theta)
        W = torch.cat((Wones,W1,W2),0)
        WVar.append(W.view(1,-1))
    D = torch.cat((WVar),0)
    if NORM:
        # Dictionary Columns Normalization except the first column
        D[:,1:] = D[:,1:]/torch.linalg.norm(D[:,1:], dim=0, keepdim=True)

    return D


class DAN(nn.Module):
    def __init__(self, 
                 N, T, lam_f,
                 wiRH=False,
                 freezeD=False):
        """
        N: int, number of poles(poles' amount is 2*N+1)
        T: int, length of data sequence
        lam: float, regulization weight for FISTA solution to LASSO
        wiRH: bool, use Reweighted Heuristic algorithm or not
        freezeD: bool, freeze the Dictionary or not
        """
        super(DAN, self).__init__()
        self.T = T # Sequence length
        self.wiRH = wiRH # With Reweighted Heuristic Algorithm or not
        self.lam_f = lam_f # Lambda for FISTA
        # Initialize Poles
        rho, theta = torch.rand(N), torch.rand(N)
        # rho, theta = torch.ones(N), torch.zeros(N)
        self.rho, self.theta = nn.Parameter(rho), nn.Parameter(theta)
        self.freezeD = freezeD
        # Frozen Dictionary, no need to create each forward
        if freezeD:  self.self_D()

    # def init_P(N=81):
    #     """ Grid initialization rather than 
    #     """
    #     n_grid = int(np.sqrt(N))
    #     rho = np.arange()
    #     theta = np.arange()
        
    def init_P_DYAN(self, N):
        """Initialize the poles without pretrained ones
        """
        eps_l = 0.25 # 0.15
        eps_h = 0.15
        rmin = (0.90 - eps_l)
        rmax = (0.90 + eps_h)
        rmin2 = pow(rmin, 2)
        rmax2 = pow(rmax, 2)
        delta = 0.001
        xv = np.arange(-rmax, rmax, delta)
        x, y = np.meshgrid(xv, xv, sparse=False)
        thetaMin = 0.001
        thetaMax = np.pi - 0.001
        mask = np.logical_and(np.logical_and(x ** 2 + y ** 2 >= rmin2, x ** 2 + y ** 2 <= rmax2),
                            np.logical_and(np.angle(x + 1j * y) >= thetaMin, np.angle(x + 1j * y) <= thetaMax))
        px, py = x[mask], y[mask]
        Pool_poles = px + 1j * py
        
        N_all = len(Pool_poles)
        Npole = int(N/2)
        idx = random.sample(range(0, N_all), Npole)
        P = Pool_poles[idx]
        
        rho = torch.from_numpy(abs(P)).float()
        theta = torch.from_numpy(np.angle(P)).float()

        return rho, theta

        
    # def init_P_021(self, N):
    #     """Initialize the poles without pretrained ones
    #     INPUT:
    #         N: int, number of conjugate pole pairs
    #     """
    #     return torch.rand(N), torch.rand(N)

    def self_D(self):
        """Create the Dictionary if the rho and theta are fixed
        """
        self.D = real_D(self.T, self.rho, self.theta)
        # self.D = D/torch.norm(D,dim=0, keepdim=True)
        # self.D = D/torch.linalg.norm(D, dim=0, keepdim=True)
        self.D[:,1:] = self.D[:,1:]/torch.linalg.norm(self.D[:,1:], dim=0, keepdim=True)


    def FISTA(self, x, dic):
        """ Return FISTA Optimized Solution and Reconstruction
        """
        C = fista_efc(dic, x, self.lam, 100)
        # Reconstruction
        reconst = torch.matmul(dic, C)

        return C, dic, reconst

    def RH_FISTA(self, x, dic):
        '''With Reweighted Heuristic Algorithm
        x: N x T x (num_j x dim_j)
        '''
        # if self.freezeD: 
        #     D = self.D.detach() # Detach to ensure it's not part of a computation graph
        # else:            
        #     D = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
        #     ipdb.set_trace()
        #     D = D/torch.norm(D,dim=0, keepdim=True)
        #     ipdb.set_trace()
        # batch_size, num_poles, num_joints x dim_joint
        Nsample, Npole, Ddata = x.shape[0], dic.shape[1], x.shape[2]
        w_init = torch.ones(Nsample, Npole, Ddata)

        i = 0
        while i < 2:
            # temp: [N, Np, num_j x dim_j]
            temp = fista_reweighted(dic, x, self.lam, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-2)
            # ipdb.set_trace()
            # Scaler for Reweight
            # Batch-wise
            # w_init = (w/torch.norm(w)) * D.shape[1]
            # Matrix-wise
            # w_init = (w/torch.norm(w, dim=(1,2)).view(-1,1,1)) * Npole
            # Column-wise for Coefficients
            # w_init = (w/torch.norm(w, dim=(1)).view(Nsample,1,-1)) * Npole # norm 2
            # w_init = (w/torch.norm(w, p=1, dim=(1)).view(Nsample,1,-1)) * Npole # norm 1, pytorch<1.9.0
            w_init = (w/torch.linalg.norm(w, ord=1, dim=1, keepdim=True)) * Npole # norm 1, pytorch>=1.9.0

            
            final = temp
            del temp
            i += 1

        sparseCode = final
        reconst = torch.matmul(dic, sparseCode.cuda(self.gpu_id))

        return sparseCode, dic, reconst
    
    def tensor_hook1(self, grad):
        print(f"Gradient of rr: {grad}")
        ipdb.set_trace()

    def tensor_hook2(self, grad):
        print(f"Gradient of theta: {grad}")
        ipdb.set_trace()

    def forward(self, x):
        # print('rr: max', self.rr.max(),'min',self.rr.min(),'mean', self.rr.mean())
        # print('theta: max', self.theta.max(), 'min', self.theta.min(), 'mean',self.theta.mean())
        # self.rr.register_hook(self.tensor_hook1)
        # self.theta.register_hook(self.tensor_hook2)
        dic = self.D.detach() if self.freezeD else real_D(self.T, self.rho, self.theta)
        # dic = dic/torch.linalg.norm(dic, dim=0, keepdim=True)
        if self.wiRH:   return self.RH_FISTA(x, dic)
        else:           return self.FISTA(x, dic)