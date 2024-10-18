# DIR source code for CVAR

## Environment
- python=3.7
- PyTorch=1.6.0

## Dataset:
[Northwestern-UCLA Multiview Action 3D Dataset(NUCLA)](https://wangjiangb.github.io/my_data.html)
[List]() for Cross View setup.

## Training:
Almost all hyperparameters included as default values(Now only support **NUCLA**/**Cross-View**/**Single**/**Gumbel**/**Re-Weighted DYAN**(RHdyan)). \
Pretrained Dictionary for step 1: [link](https://northeastern-my.sharepoint.com/:u:/r/personal/luo_dan1_northeastern_edu/Documents/Mine/2021-CrossView/src_code/202410_Journal/pretrained/yuexi_NUCLA_CV_Single_rhDYAN_bi_100.pth?csf=1&web=1&e=4OUaI2)

1. DIR-Dictionary(1145MiB GRAM)

    `bash    
        python train_DIR_D_NUCLA.py --modelRoot /dir/to/save --path_list /dir/to/data
    `
2. DIR-Classification(1107MiB GRAM)

    `bash    
        python train_DIR_cls_noCL_NUCLA.py --modelRoot /dir/to/save --path_list /dir/to/data --pretrain /path/to/model
    `
3. DIR-Contrastive Learning

    `bash    
        python train_DIR_cls_wiCL_NUCLA.py --modelRoot /dir/to/save --path_list /dir/to/data --pretrain /path/to/model
    `
4. DIR-Finetune(1023MiB GRAM)

    `bash    
        python train_DIR_cls_ft_NUCLA.py --modelRoot /dir/to/save --path_list /dir/to/data --pretrain /path/to/model
    `