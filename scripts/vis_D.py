''' Visualize the movement of trained poles on the Polar,
    default rho and theta are the input of the sigmoid function
    25/01, Dan
'''
import os
import sys
import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt

import torch

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


def main(args):
    '''
    INPUT:
        sys.argv[1]: name of the generated gif
        sys.argv[2]: directory saving all trained models 
    '''
    name_gif = args[1]
    dir_pth = args[2]
    print("Generating Dictionary Poles change visualizaiton:", name_gif+'.gif')
    print("Loading models from:", dir_pth)
    list_name = glob.glob(dir_pth+"/*.pth")
    list_name.sort(key=sortKey)

    frames = []
    r_0 = 0
    theta_0 = 0
    for i, path_pth in enumerate(list_name):
        name_pth = path_pth.split('/')[-1]
        state_dict = torch.load(path_pth, map_location='cpu')['state_dict']
        r, theta = num2rt(state_dict['sparseCoding.rho'], state_dict['sparseCoding.theta']) # rmax=3.0
        r, theta = r.numpy(), theta.numpy()
        
        ax = plt.subplot(1, 1, 1, projection='polar')
        if name_pth == 'dir_d_0.pth':
            r_0 = r
            theta_0 = theta
            
            ax.scatter(0, 1, c='red', s = 5)
        else:
            ax.scatter(0, 1, c='green', s = 5)
            ax.scatter( theta, r, c='green', s = 5)
            ax.scatter(-theta, r, c='green', s = 5)
        if i == len(list_name)-1:
            ax.set_rmax(1.2)
            ax.set_title(f"Dictionary @Ep {name_pth.split('.')[0].replace('dir_d_','')}", va='top', pad=20)
            plt.draw()
            plt.savefig(os.path.join(dir_pth, f'Dict_epoch{i}.png'))
        ax.scatter( theta_0, r_0, c='red', s = 5)
        ax.scatter(-theta_0, r_0, c='red', s = 5)
        ax.set_rmax(1.2) # 1.2 3.1

        ax.set_title(f"Dictionary @Epoch {name_pth.split('.')[0]}", va='top', pad=20)
        plt.draw()
        if (i == 0):
            plt.savefig(os.path.join(dir_pth, f'Dict_epoch{i}.png'))

        frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
    
    # Save frames as .gif using imageio
    imageio.mimsave(os.path.join(dir_pth, name_gif+'.gif'),
                    frames, fps=3)
    print(name_gif+'.gif','saved under:', dir_pth)

def main_clamp(args):
    '''
    INPUT:
        sys.argv[1]: name of the generated gif
        sys.argv[2]: directory saving all trained models 
    '''
    name_gif = args[1]
    dir_pth = args[2]
    print("Generating Dictionary Poles change visualizaiton:", name_gif+'.gif')
    print("Loading models from:", dir_pth)
    list_name = glob.glob(dir_pth+"/*.pth")
    list_name.sort(key=sortKey)

    frames = []
    r_0 = 0
    theta_0 = 0
    for i, path_pth in enumerate(list_name):
        name_pth = path_pth.split('/')[-1]
        state_dict = torch.load(path_pth, map_location='cpu')['state_dict']
        r, theta = num2rt_clamp(state_dict['sparseCoding.rho'], state_dict['sparseCoding.theta'],
                                0.8,1.1, 0,torch.pi) # rmax=3.0
        r, theta = r.numpy(), theta.numpy()
        
        ax = plt.subplot(1, 1, 1, projection='polar')
        if name_pth == 'dir_d_0.pth':
            r_0 = r
            theta_0 = theta
            
            ax.scatter(0, 1, c='red', s = 5)
        else:
            ax.scatter(0, 1, c='green', s = 5)
            ax.scatter( theta, r, c='green', s = 5)
            ax.scatter(-theta, r, c='green', s = 5)
        if i == len(list_name)-1:
            ax.set_rmax(1.2)
            ax.set_title(f"D @Ep {name_pth.split('.')[0].replace('dir_d_','')}", va='top', pad=20)
            plt.draw()
            plt.savefig(os.path.join(dir_pth, f'D_epoch{i}.png'))
        ax.scatter( theta_0, r_0, c='red', s = 5)
        ax.scatter(-theta_0, r_0, c='red', s = 5)
        ax.set_rmax(1.2) # 1.2 3.1

        ax.set_title(f"D @Ep {name_pth.split('.')[0]}", va='top', pad=20)
        plt.draw()
        if (i == 0):
            plt.savefig(os.path.join(dir_pth, f'D_epoch{i}.png'))

        frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
    
    # Save frames as .gif using imageio
    imageio.mimsave(os.path.join(dir_pth, name_gif+'.gif'),
                    frames, fps=3)
    print(name_gif+'.gif','saved under:', dir_pth)

def main_cvar(args):
    '''
    INPUT:
        sys.argv[1]: name of the generated gif
        sys.argv[2]: directory saving all trained models 
    '''
    name_gif = args[1]
    # dir_pth = args[2]
    dir_pth = "/home/dan/ws/202209_CrossView/202412-CVAR_CL/pretrained/NUCLA/setup1/Multi/pretrainedRHDYAN_for_CL.pth"
    print("Generating Dictionary Poles change visualizaiton:", name_gif+'.gif')
    print("Loading models from:", dir_pth)
    list_name = glob.glob(dir_pth)

    def sortKey(s):
        return int(s.split('/')[-1].split('.')[0].replace('pretrainedRHDYAN_for_CL','0'))
    list_name.sort(key=sortKey)

    frames = []
    r_0 = 0
    theta_0 = 0
    for i, path_pth in enumerate(list_name):
        name_pth = path_pth.split('/')[-1]
        state_dict = torch.load(path_pth, map_location='cpu')['state_dict']
        r = state_dict['backbone.sparseCoding.rr'].numpy()
        theta = state_dict['backbone.sparseCoding.theta'].numpy()
        
        ax = plt.subplot(1, 1, 1, projection='polar')
        if i==0:
            r_0 = r
            theta_0 = theta
            
            ax.scatter(0, 1, c='red', s = 5)
        else:
            ax.scatter(0, 1, c='green', s = 5)
            ax.scatter( theta, r, c='green', s = 5)
            ax.scatter(-theta, r, c='green', s = 5)
        if i == len(list_name)-1:
            ax.set_rmax(1.2)
            ax.set_title(f"D cvar", va='top')
            plt.draw()
            plt.savefig(f'Dict_cvar_cl.png')
        ax.scatter( theta_0, r_0, c='red', s = 5)
        ax.scatter(-theta_0, r_0, c='red', s = 5)
        ax.set_rmax(1.2)
        ax.set_title(f"D cvar", va='top',pad=20)
        plt.draw()
        if (i == 0):
            plt.savefig(f'Dict_cvar.png')

        frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
    
    # Save frames as .gif using imageio
    imageio.mimsave(name_gif+'.gif',
                    frames, fps=3)
    print(name_gif+'.gif', 'saved under:', dir_pth)


if __name__ == '__main__':
    # main(sys.argv)
    # main_cvar(sys.argv)
    main_clamp(sys.argv)