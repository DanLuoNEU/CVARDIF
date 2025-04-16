############################# Import Section #################################
import sys
from math import sqrt
import numpy as np

import torch
import torch.nn as nn
# sys.path.append('../')
# sys.path.append('../data')
# sys.path.append('.')
from utils import ipdb, random, num2rt, rt2num, num2rt_clamp, rt2num_clamp
from modelZoo.gumbel_module import GumbelSigmoid
############################# Import Section #################################
def fista(D, Y, lam_f, maxIter=100):
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
    Softshrink = nn.Softshrink(lam_f*L_inv.item())
    
    x = torch.zeros(num_pole, dim_data, device=device)
    y = torch.zeros(num_pole, dim_data, device=device)
    t = 1

    # del DtD
    for _ in range(maxIter):
        # Gradient step
        x_new = Softshrink( A @ y + B)
        # Check convergence
        if torch.linalg.norm((x - x_new))/dim_data < 1e-5:
            # del A, B

            return x_new
        # Momentum update
        t_new = (1. + np.sqrt(1 + 4 * t**2)) / 2.
        tt = ( t - 1 ) / t_new
        y_new = (1 + tt) * x_new - tt * x
        # Update variables
        x = x_new
        t = t_new
        y = y_new
    # del A, B

    return x


def fista_reweighted(D, Y, lambd, w, maxIter):
    '''
        D: [T, 161]\\
        Y: [Nsample, T, 50]\\
        w: [Nsample, 161, 25 x 2]
    '''
    if len(D.shape) < 3:
        DtD = D.T @ D # torch.matmul(torch.t(D), D)
        DtY = D.T @ Y # torch.matmul(torch.t(D), Y)
    else:
        DtD = D.permute(0, 2, 1) @ D # torch.matmul(D.permute(0, 2, 1), D)
        DtY = D.permute(0, 2, 1) @ Y # torch.matmul(D.permute(0, 2, 1), Y)
    # Spectral norm/largest singular value of DtD
    # # Option 1: PyTorch < 1.9.0
    # L = spectral_norm_svd(DtD)
    # Option 2: PyTorch >=1.9.0
    # L = torch.linalg.norm(DtD, ord=2)
    try:
        L = torch.linalg.norm(DtD, ord=2)  # This may fail
    except RuntimeError as e:
        print("Error occurred:", e)
        ipdb.set_trace()

    # # Here uses a highly optimized algorithm to estimate the largest singular value instead
    # # Ensure DtD is not rank-deficient or ill-conditioned, otherwise, this can lead to differences in results
    # L = torch.norm(DtD, 2) # element-wise norm
    Linv = 1/L
    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).to(D.device) - DtD * Linv
    const_xminus = DtY * Linv - weightedLambd.to(D.device)
    const_xplus =  DtY * Linv + weightedLambd.to(D.device)

    t_old = 1
    iter = 0
    while iter < maxIter:
        iter +=1
        Ay = A @ y_old
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old-1)/t_new
        y_new = x_new + tt * (x_new-x_old)  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        # if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
        if torch.linalg.norm((x_old - x_new))/x_old.shape[1] < 1e-5:
            # x_old = x_new
            # break
            
            return x_new
        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old


def real_D(T, rho, theta, 
           CONSTR=True,
           rmin=0.001, rmax=1.1,
           tmin=0., tmax=torch.pi,
           # rmin=0.8415, rmax=1.0852,
           # tmin=0.1687, tmax=3.1052,
           NORM=False):
    """ Create Real Dictionary
    INPUT:
        T:int, sequence length
        rho:torch.tensor(float), (N)
        theta:torch.tensor(float),(N)
    """
    if CONSTR:
        # rho, theta = num2rt(rho, theta,
        #                     rmin, rmax, tmin, tmax)
        rho, theta = num2rt_clamp(rho, theta,
                                  rmin, rmax, tmin, tmax)
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
        D = D/torch.linalg.norm(D, dim=0, keepdim=True)

    return D


class DYAN(nn.Module):
    """Binarized Sparse Coding
    """
    def __init__(self, 
                 args, freezeD=False):
        """
        N: int, number of poles(poles' amount is 2*N+1)
        T: int, length of data sequence
        lam: float, regulization weight for FISTA solution to LASSO
        wiRH: bool, use Reweighted Heuristic algorithm or not
        freezeD: bool, freeze the Dictionary or not
        """
        super(DYAN, self).__init__()
        self.N = args.N
        self.T = args.T # Sequence length
        self.lam_f = args.lam_f # Lambda for FISTA
        self.rmin = float(args.r_r.split(',')[0])
        self.rmax = float(args.r_r.split(',')[1]) # range for rho
        self.tmin = float(args.r_t.split(',')[0])
        self.tmax = float(args.r_t.split(',')[1]) if args.r_t.split(',')[1]!='pi' else torch.pi # range for theta

        self.wiRH = args.wiRH # With Reweighted Heuristic Algorithm or not
        self.freezeD = freezeD # Freeze Dictionary or not, NOTE: no need to create again for each forward()

        # Initialize Poles
        ## Different Init quadrant
        # Option 1. 2nd quad
        ## NOTE: rho and theta are in range [0,1], so initialized poles are in 2nd quad without any reparameterization
        # print('2nd quadrant')
        # self.rho, self.theta = nn.Parameter(torch.rand(N)), nn.Parameter(torch.rand(N))
        # Option 2. 1st+2nd quad init
        # print('Poles Initialization: 1st + 2nd quadrants')
        # self.rho, self.theta = nn.Parameter((torch.rand(self.N)- 0.5) * 4), nn.Parameter((torch.rand(self.N)- 0.5) * 4)
        # Option 3. 1st quad init
        # print('Poles Initialization: 1st quadrant')
        # self.rho, self.theta = nn.Parameter(torch.rand(N)- 1.0), nn.Parameter(torch.rand(N)- 1.0) 
        # Option 4. Uniformly distributed on the unit circle
        # print('Poles Initialization: uniformally distributed')
        # self.rho, self.theta = rt2num(torch.ones(self.N), torch.linspace(0, torch.pi, self.N))
        # Option 5. Finetuning CVAR Pretrained poles
        # print('Poles Initialization: cvar pretrained')
        # path_cvar_pret = "/home/dan/ws/202209_CrossView/202412-CVAR_CL/pretrained/NUCLA/setup1/Multi/pretrainedRHDYAN_for_CL.pth"
        # print('   cvar pretrained:', path_cvar_pret)
        # state_dict=torch.load(path_cvar_pret, map_location='cpu')['state_dict']
        # self.rho, self.theta = rt2num(state_dict['backbone.sparseCoding.rr'], state_dict['backbone.sparseCoding.theta'],
        #                               self.rmin, self.rmax, self.tmin, self.tmax)
        # # Option 6.Initialize poles around CVAR pole ranges
        # print('Poles Initialization: cvar pretrained range') # 0.1687, tmax=3.1052
        # self.rho, self.theta = rt2num(torch.rand(self.N)*(1.1-0.8)+0.8,
        #                               torch.rand(self.N)*(3.11-0.16)+0.16,
        #                               self.rmin, self.rmax, self.tmin, self.tmax)
        # Option 7.Initialize poles around CVAR pole ranges without limit
        # print('Poles Initialization: cvar pretrained range, trained without limit range')
        # self.rho, self.theta = torch.rand(self.N)*(1.1-0.8)+0.8, torch.rand(self.N)*(3.11-0.16)+0.16
        # # Option 8.Initialize poles around CVAR pole ranges without limit
        # print('Poles Initialization: cvar pretrained range, clamp') 
        self.rho, self.theta = rt2num_clamp(torch.rand(self.N)*(self.rmax-self.rmin)+self.rmin,
                                            torch.rand(self.N)*(self.tmax-self.tmin)+self.tmin,
                                            self.rmin, self.rmax, self.tmin, self.tmax)
        
        # Option 9. 1st+2nd quad init
        # print('Poles Initialization: 1st + 2nd quadrants, clamp')
        # self.rho, self.theta = torch.rand(self.N), torch.rand(self.N)


        self.rho, self.theta = nn.Parameter(self.rho), nn.Parameter(self.theta)


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


    def update_D(self, rho, theta):
        """Create the Dictionary if the rho and theta are fixed
        """
        device = self.rho.device
        self.rho = nn.Parameter(rho.to(device))
        self.theta = nn.Parameter(theta.to(device))
        self.D = real_D(self.T, self.rho, self.theta, True,
                        self.rmin, self.rmax, self.tmin, self.tmax)
        # # Column normalization for Dictionary
        # self.D = self.D/torch.linalg.norm(self.D, dim=0, keepdim=True)


    def FISTA(self, x, D):
        """ Return FISTA Optimized Solution and Reconstruction
        """
        C = fista(D, x, self.lam_f, 100)
        # Reconstruction
        R = D @ C

        return C, D, R


    def RH_FISTA(self, x, D):
        '''With Reweighted Heuristic Algorithm
        x: [(B x C), T, (num_j x dim_j)]
        D: [T, Np]
        '''
        # if self.freezeD: 
        #     D = self.D.detach() # Detach to ensure it's not part of a computation graph
        # else:            
        #     D = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
        # batch_size x num_clips, num_poles, num_joints x dim_joint
        Nsample, Npole, Ddata = x.shape[0], D.shape[1], x.shape[2]
        w_init = torch.ones(Nsample, Npole, Ddata)

        i = 0
        while i < 2:
            # temp: [N, Np, num_j x dim_j]
            temp = fista_reweighted(D, x, self.lam_f, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-2)
            # ipdb.set_trace()
            # Scaler for Reweight, Column-wise for Coefficients
            # w_init = (w/torch.norm(w, p=1, dim=(1)).view(Nsample,1,-1)) * Npole # norm 1, pytorch<1.9.0
            w_init = (w/torch.linalg.norm(w, ord=1, dim=1, keepdim=True)) * Npole # norm 1, pytorch>=1.9.0
            
            # final = temp
            # del temp
            i += 1

        C = temp # final
        R = D @ C

        return C, D, R
    

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
        dic = self.D.detach() if self.freezeD else real_D(self.T, self.rho, self.theta, True,
                                                          self.rmin, self.rmax, self.tmin, self.tmax).to(x)
        # dic = dic/torch.linalg.norm(dic, dim=0, keepdim=True)
        if self.wiRH:   return self.RH_FISTA(x, dic)
        else:           return self.FISTA(x, dic)


class BiSC(nn.Module):
    def __init__(self,
                 args):
        super(BiSC, self).__init__()
        self.sparseCoding = DYAN(args,
                                 freezeD=False)
        # Binary Module
        self.wiBI = args.wiBI
        if self.wiBI:
            self.g_th = args.g_th
            self.g_te = args.g_te
            self.BinaryCoding = GumbelSigmoid()

    def tensor_hook1(self, grad):
        print(f"Gradient of C: {grad}")

    def tensor_hook2(self, grad):
        print(f"Gradient of D: {grad}")
    
    def tensor_hook3(self, grad):
        print(f"Gradient of B: {grad}")

    def forward(self, x, inference=True):
        # DYAN encoding
        C, D, R = self.sparseCoding(x)
        # C.register_hook(self.tensor_hook1)
        # D.register_hook(self.tensor_hook2)
        # R = D@C
        # ipdb.set_trace()
        if self.wiBI:
            # Gumbel    
            B = self.BinaryCoding(C**2, self.g_th, temperature=self.g_te,
                                  force_hard=True, inference=inference)
        else:
            B = torch.ones_like(C).to(C)
        # B.register_hook(self.tensor_hook3)
        C_B = C * B
        R_B = D @ C_B

        return C, R, B, R_B

class YNet(nn.Module):
    def __init__(self, num_class, Npole, dataType, useCL):
        super(YNet, self).__init__()
        self.num_class = num_class
        self.Npole = 2 * Npole + 1
    
        self.useCL = useCL
        self.dataType = dataType
        self.conv1 = nn.Conv1d(self.Npole, 256, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=256, eps=1e-05, affine=True)

        self.conv2 = nn.Conv1d(256, 512, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=512, eps=1e-5, affine=True)

        self.conv3 = nn.Conv1d(512, 1024, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=1024, eps=1e-5, affine=True)
       
        self.conv4 = nn.Conv2d(self.Npole + 1024, 1024, (3, 1), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=1024, eps=1e-5, affine=True)

        self.conv5 = nn.Conv2d(1024, 512, (3, 1), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-5, affine=True)

        self.conv6 = nn.Conv2d(512, 256, (3, 3), stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=256, eps=1e-5, affine=True)
        if self.dataType == '2D' :
            self.njts = 25
            self.fc = nn.Linear(256*10*2, 1024) #njts = 25
        elif self.dataType == 'rgb':
            self.njts = 512 # for rgb
            # self.fc = nn.Linear(256*61*2, 1024) #for rgb
            self.fc = nn.Linear(256*253*2, 1024) # for att rgb
        elif self.dataType == '2D+rgb':
            self.njts = 512+25
            self.fc = nn.Linear(256*266*2,1024)
            
        self.pool = nn.AvgPool1d(kernel_size=(self.njts))
        # self.fc = nn.Linear(7168,1024) #njts = 34
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

        # self.linear = nn.Sequential(nn.Linear(256*10*2,1024),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(1024,512),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(512, 128),
        #                             nn.LeakyReLU())
        if self.useCL == False:
            # self.cls = nn.Linear(128, self.num_class)
            self.cls = nn.Sequential(nn.Linear(128,128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, self.num_class))
        else:
            self.cls = nn.Sequential(nn.Linear(128, self.num_class))
        self.relu = nn.LeakyReLU()

        'initialize model weights'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu' )
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self,x):
        inp = x
        if self.dataType == '2D' or 'rgb':
            dim = 2
        else:
            dim = 3

        bz = inp.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # ipdb.set_trace()
        x_gl = self.pool(self.relu(self.bn3(self.conv3(x))))
        # ipdb.set_trace()
        x_new = torch.cat((x_gl.repeat(1,1,inp.shape[-1]),inp),1).reshape(bz,1024+self.Npole, self.njts,dim)

        x_out = self.relu(self.bn4(self.conv4(x_new)))
        x_out = self.relu(self.bn5(self.conv5(x_out)))
        x_out = self.relu(self.bn6(self.conv6(x_out)))

        'MLP'
        # ipdb.set_trace()
        x_out = x_out.view(bz,-1)  #flatten
        
        x_out = self.relu(self.fc(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out)) #last feature before cls

        out = self.cls(x_out)

        return out, x_out

class CVARDIF(nn.Module):
    def __init__(self, N=80, T=36, lam_f=0.1,
                 wiRH=True,
                 wiBI=True, g_th=0.505, g_te=0.1,
                 num_class=10,
                 dataType='2D', useCL=False):
        super(CVARDIF, self).__init__()
        # DYAN
        self.T = T
        self.lam_f = lam_f
        self.wiBI = wiBI
        # Gumbel
        # self.bi_thresh = 0.505
        # Classifier
        self.num_class = num_class
        self.Npole = N
        self.dataType = dataType
        self.useCL = useCL
        # Networks
        self.sparseCoding = DYAN( N, T, lam_f,
                                  wiRH,
                                  freezeD=True)
        if self.wiBI:
            self.g_th = g_th
            self.g_te = g_te
            self.BinaryCoding = GumbelSigmoid()
        self.Classifier = YNet(num_class=self.num_class,
                                Npole=self.Npole,
                                dataType=self.dataType,
                                useCL=self.useCL)

    def forward(self, x, g_inf=False):
        '''
            g_inf: Gumbel Inference, train-False, val/test-True
        '''
        # DYAN
        C, D, R_C = self.sparseCoding(x) # w.RH
        if self.wiBI:
            # Gumbel
            B = self.BinaryCoding(C**2, self.g_th, temperature=self.g_te,
                                        force_hard=True, inference=g_inf)
            # Classifier
            label, lastFeat = self.Classifier(B)
        else:
            B = torch.zeros_like(C).to(C)
            label, lastFeat = self.Classifier(C)
        C_B = C * B
        R_B = torch.matmul(D, C_B)

        return label, lastFeat, C, R_C, B, R_B