''' Classification depending on Action-wise Binary Map based on synthetic data
    25/03/21

    # 1) Binary templates for different classes
    # 2) generate samples based on those templates
    # 	1000 samples based on templates, among those flip 2% to the opposite
    # 	train/test 9:1
    # 3) pass through classifier
    # 	- baseline - max XNOR with templates
    # 	- MLP - BiMLP
'''

import os
import sys
import glob
import ipdb
import numpy as np
from einops import repeat, rearrange
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from torch.utils.data import Dataset, DataLoader

# from modelZoo.networks import BiSC
# from utils import sparsity

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class SYN_B(Dataset):
    """
    """
    def __init__(self, phase='train',
                 num_cls=3, num_poles = 161, dim_data=10,
                 num_samples=1000, rate_flip=0.10):
        data_load = False
        if os.path.exists(f'data/synB/{phase}_rf{rate_flip*100:02.0f}.pt'): 
            data = torch.load(f'data/synB/{phase}_rf{rate_flip*100:02.0f}.pt')
            assert phase==data['phase'],'Phase NOT MATCH'
            assert rate_flip==data['rate_flip'],'rate_flip NOT MATCH'
            self.B_templates = data['B_templates']
            self.B_samples = data['B_samples']
            self.label_samples = data['label_samples']
            data_load = True
        if not data_load:
            # generate Binary templates for 3 different classes
            self.B_templates, self.B_samples, self.label_samples = self.gen_B_t(num_cls, num_poles, dim_data,
                                                                            num_samples, rate_flip)
            # different trainset
            amt_train = int(0.9*num_samples)
            indices_uniq = torch.argsort(torch.rand(num_cls, num_samples), dim=1)
            indices_uniq = torch.stack([indices_uniq[i_cls,:]+i_cls*num_samples for i_cls in range(indices_uniq.shape[0])])
            indices_train = rearrange(indices_uniq[:, :amt_train], 'c s -> (c s)')
            indices_test  = rearrange(indices_uniq[:, amt_train:], 'c s -> (c s)')
            # save and 
            torch.save({'phase':         'train',
                        'rate_flip':     rate_flip,
                        'B_templates':   self.B_templates,
                        'B_samples':     self.B_samples[indices_train],
                        'label_samples': self.label_samples[indices_train]},
                        f'data/synB/train_rf{rate_flip*100:02.0f}.pt')
            torch.save({'phase':         'test',
                        'rate_flip':     rate_flip,
                        'B_templates':   self.B_templates,
                        'B_samples':     self.B_samples[indices_test],
                        'label_samples': self.label_samples[indices_test]},
                        f'data/synB/test_rf{rate_flip*100:02.0f}.pt')
            if phase=='train':
                self.B_samples = self.B_samples[indices_train]
                self.label_samples = self.label_samples[indices_train]
            else:
                self.B_samples = self.B_samples[indices_test]
                self.label_samples = self.label_samples[indices_test]

        # ipdb.set_trace()
        pass

    def gen_B_t(self, num_cls, num_poles, dim_data,
                num_samples,rate_flip):
        """Generate B templates with random sparse indices for each data dimension  

        example:  
        ipdb> xnor0=torch.sum(templates_B[0]==templates_B[1])/templates_B[0].numel()  
        ipdb> xnor0  
        tensor(0.9112)  
        ipdb> xnor0=torch.sum(templates_B[0]==0)/templates_B[0].numel()  
        ipdb> xnor0  
        tensor(0.9553)  
        ipdb> xnor1=torch.sum(templates_B[0]==templates_B[2])/templates_B[0].numel()  
        ipdb> xnor2=torch.sum(templates_B[1]==templates_B[2])/templates_B[0].numel()  
        ipdb> xnor1,xnor2  
        (tensor(0.9093), tensor(0.9037))  
        """
        B_templates = torch.zeros(num_cls, num_poles, dim_data)  # Initialize with zeros
        for i in range(num_cls):
            for j in range(dim_data):
                num_nonzero = torch.randint(5, 11, (1,))  # Choose a random number of nonzero elements
                indices = torch.randperm(num_poles)[:num_nonzero]  # Random indices
                B_templates[i, indices, j] = 1  # Assign values to random positions
        
        # generate samples based on templates
        #  - 1000 samples based on templates, among those flip 2% to the opposite
        #  - train/test 9:1
        ## duplicate templates to samples
        B_samples = repeat(B_templates, f'c p d -> (c {num_samples}) p d')
        ##  Generate random indices
        numel_template = num_poles * dim_data
        numel_flip = int(rate_flip * numel_template)
        ids_batch = torch.arange(num_cls * num_samples).unsqueeze(1).expand(-1, numel_flip)
        # indices = torch.randint(0, numel_template, (num_cls * num_samples, numel_flip)) # WRONG way to pick unique indices from a range
        indices = torch.argsort(torch.rand(num_cls * num_samples, numel_template), dim=1)[:, :numel_flip]
        ## (deprecated)
        # ids_row = indices // dim_data
        ids_row = torch.div(indices, dim_data, rounding_mode='floor')
        ids_col = indices % dim_data
        B_samples[ids_batch, ids_row, ids_col] = 1 - B_samples[ids_batch, ids_row, ids_col]
        ## GroundTruth labels
        label_samples = repeat(torch.arange(num_cls), f'c -> (c {num_samples})')
        
        # samples = rearrange(samples, '(c s) p d -> c s p d', c=num_cls)
        
        return B_templates, B_samples, label_samples
        

    def __len__(self):
      return len(self.label_samples)
    

    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """

        return self.B_samples[index], self.label_samples[index]


def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='SBAR-B-CLS')

    parser.add_argument('--num_cls', default=3, type=int, help='')
    parser.add_argument('--num_poles', default=161, type=int, help='')
    parser.add_argument('--dim_data', default=10, type=int, help='')
    parser.add_argument('--rate_flip', default=0.10, type=float, help='')

    parser.add_argument('--mode', default='data', help='data | baseline | MLP')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--nw', default=8, type=int, help='number of workers on cpu')
    parser.add_argument('--gpu_id', default=7, type=int, help='gpu id for cuda')


    return parser


def test_data(args):
    trainset = SYN_B('train', rate_flip=args.rate_flip)
    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=64, num_workers=8)
    testset = SYN_B('test', rate_flip=args.rate_flip)
    testloader = DataLoader(testset, shuffle=False,
                             batch_size=64, num_workers=8)
    for B_samples,label_samples in testloader:
        for i, B_sample in enumerate(B_samples):
            print(f'sp: {torch.sum(B_sample==0)/B_sample.numel()}',
                  f'label_gt: {label_samples[i]}',
                  f'xnor:', [f"{torch.sum((B_sample==template)/B_sample.numel()):.4f}" for template in trainset.B_templates])
        ipdb.set_trace()
    # pass
    

def cls_baseline(args):
    trainset = SYN_B('train', rate_flip=args.rate_flip)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=64, num_workers=8)
    testset = SYN_B('test', rate_flip=args.rate_flip)
    testloader = DataLoader(testset, shuffle=False, batch_size=64, num_workers=8)
    count, amount = 0, 0
    for B_samples,label_samples in testloader:
        label_xnor = torch.argmax(
                        torch.sum((B_samples.unsqueeze(1)==trainset.B_templates.unsqueeze(0)),
                                    dim=(2,3)),
                                    dim=1)
        count += torch.sum(label_xnor==label_samples)
        amount += label_samples.shape[0]
    
    # print(f'rate_flip: {args.rate_flip} tlest acc: {count/amount:.4f}')
    
    return count/amount


def cls_stat(args):
    # DataLoader
    trainset = SYN_B('train', rate_flip=args.rate_flip)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=64, num_workers=8)
    testset = SYN_B('test', rate_flip=args.rate_flip)
    testloader = DataLoader(testset, shuffle=False, batch_size=64, num_workers=8)
    # Train
    B_stat = torch.zeros(args.num_cls, args.num_poles, args.dim_data,dtype=torch.float32)
    n_samples = torch.zeros(args.num_cls,dtype=torch.float32)
    for B_samples, label_samples in trainloader:
        n_samples.scatter_add_(0, label_samples, torch.ones_like(label_samples,dtype=torch.float32))
        # B_stat.scatter_add_(0, label_samples.view(-1, 1, 1
        #                                 ).expand(-1, B_samples.shape[1], B_samples.shape[2]
        #                         ), B_samples)
        for i in range(label_samples.shape[0]):  # Iterate over batch
            B_stat[label_samples[i]] += B_samples[i]
    # ipdb.set_trace()
    B_stat = B_stat/n_samples.view(-1,1,1)
    
    # Test
    ths_stat = [th_s/100 for th_s in range(101)]
    list_acc = []
    for th_stat in ths_stat:
        B_templates = (B_stat>th_stat).int()
        count, amount = 0, 0
        for B_samples,label_samples in testloader:
            label_xnor = torch.argmax(
                            torch.sum((B_samples.unsqueeze(1)==B_templates.unsqueeze(0)),
                                        dim=(2,3)),
                                        dim=1)
            count += torch.sum(label_xnor==label_samples)
            amount += label_samples.shape[0]
        
        list_acc.append(count/amount)
        print(f'rate_flip: {args.rate_flip} threshold: {th_stat} acc: {count/amount:.4f}')
    
    return list_acc


def sim_stat(args):
    # DataLoader
    trainset = SYN_B('train', rate_flip=args.rate_flip)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=64, num_workers=8)
    # Train
    B_stat = torch.zeros(args.num_cls, args.num_poles, args.dim_data,dtype=torch.float32)
    n_samples = torch.zeros(args.num_cls,dtype=torch.float32)
    for B_samples, label_samples in trainloader:
        n_samples.scatter_add_(0, label_samples, torch.ones_like(label_samples,dtype=torch.float32))
        # B_stat.scatter_add_(0, label_samples.view(-1, 1, 1
        #                                 ).expand(-1, B_samples.shape[1], B_samples.shape[2]
        #                         ), B_samples)
        for i in range(label_samples.shape[0]):  # Iterate over batch
            B_stat[label_samples[i]] += B_samples[i]
    # ipdb.set_trace()
    B_stat = B_stat/n_samples.view(-1,1,1)
    
    ths_stat = [th_s/100 for th_s in range(101)]
    list_sim = []
    for th_stat in ths_stat:
        B_reconst = (B_stat>th_stat).int()
        sim = (B_reconst==trainset.B_templates).int().sum()/B_reconst.numel()
        list_sim.append(sim)
        print(f'rate_flip: {args.rate_flip} threshold: {th_stat} sim: {sim:.4f}')
    
    return list_sim


def cls_MLP(args):
    trainset = SYN_B()
    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=64, num_workers=8)
    
    for B_samples,label_samples in trainloader:
        for i, B_sample in enumerate(B_samples):
            print(f'sp: {torch.sum(B_sample==0)/B_sample.numel()}',
                  f'label_template: {label_samples[i]}',
                  f'xnor:', [f"{torch.sum((B_sample==template)/B_sample.numel()):.4f}" for template in trainset.B_templates])
        ipdb.set_trace()
    pass


if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    if args.mode == 'data':
        # test dataloader
        test_data(args)
    elif args.mode == 'baseline':
        # baseline
        r_flip = [r/100 for r in range(101)]
        list_acc = []
        for r in r_flip:
            args.rate_flip = r
            acc = cls_baseline(args)
            list_acc.append(acc)
            print(f'rate_flip: {args.rate_flip} acc: {acc:.4f}')
        plt.figure(figsize=(8, 5))
        plt.plot(r_flip, list_acc, marker='o', linestyle='-', color='r', label='Accuracy')

        # Labels and title
        plt.xlabel("Rate Flip")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Rate Flip")
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig("notes_exp/20250320_syn_C/20250328_baseline_acc-wi-rate_flip.png", dpi=300, bbox_inches='tight')
    elif args.mode == 'stat_cls':
        # statistic way
        r_flip = [0.02,0.2,0.4,0.5,0.55,0.6,0.8]
        markers = ['x','o','s','.','D','*','+']
        colors = ['brown','r','b','g','c','m','y']
        x_ths = [th_s/100 for th_s in range(101)]

        plt.figure(figsize=(8, 5))
        for i_r, r in enumerate(r_flip):
            print(i_r, markers[i_r],colors[i_r])
            args.rate_flip = r
            list_acc = cls_stat(args)
            plt.plot(x_ths, list_acc, marker=markers[i_r], linestyle='-', 
                     color=colors[i_r], label=f'{args.rate_flip}')

        # Labels and title
        plt.xlabel("Threshold for B")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs th_B")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig("notes_exp/20250320_syn_C/20250328_stat_acc-wi-th-rate_flip.png", dpi=300, bbox_inches='tight')
    elif args.mode == 'stat_sim':
        # statistic way
        r_flip = [0.02,0.2,0.4,0.5,0.55,0.6,0.8]
        markers = ['x','o','s','.','D','*','+']
        colors = ['brown','r','b','g','c','m','y']
        x_ths = [th_s/100 for th_s in range(101)]

        plt.figure(figsize=(8, 5))
        for i_r, r in enumerate(r_flip):
            print(i_r, markers[i_r],colors[i_r])
            args.rate_flip = r
            list_sim = sim_stat(args)
            plt.plot(x_ths, list_sim, marker=markers[i_r], linestyle='-', 
                     color=colors[i_r], label=f'{args.rate_flip}')

        # Labels and title
        plt.xlabel("Threshold for B")
        plt.ylabel("Similarity")
        plt.title("Similarity vs th_B")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig("notes_exp/20250320_syn_C/20250331_stat_sim-wi-th-rate_flip.png", dpi=300, bbox_inches='tight')
    elif args.mode == 'MLP':
        # BiMLP
        cls_MLP(args)