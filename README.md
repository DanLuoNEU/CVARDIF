# Official Implementation for CVAR

## Environment
- python=3.7.4
- pytorch=1.10.0+cu111

## Original [Github Repo](https://northeastern-my.sharepoint.com/:u:/g/personal/luo_dan1_northeastern_edu/EdzEbgrHE-1DocUc7IdQ6-EBekyUWyDZt-wXyw5fRCQPLg?email=camps%40coe.neu.edu&e=M2cv2I) from Yuexi Zhang

## Dataset:
[Northwestern-UCLA Multiview Action 3D Dataset(NUCLA)](https://wangjiangb.github.io/my_data.html) \
[List](data/CV/setup1) for Cross View setup.

## Training:
Almost all hyperparameters included as default values(Now only support **NUCLA**/**Cross-View**/**Multi**/**Gumbel**/**Re-Weighted DYAN**(RHdyan)). 

1. DIR-Dictionary(1145MiB GRAM)

    ```bash    
        python train_DIR_D_NUCLA.py  --modelRoot /path/to/save/model_files --sampling Multi --nClip 6 --wiRH 1 --wiBI 1 --bs 32 --gpu_id 0 --lam_f 1e-2 --g_th 0.505 --lam2 5e-1 --Alpha 1e-2 --lr_2 1e-2
    ```
    Here is the corresponding [log](notes_exp/SBAR_SparseCoding/3_clamp-L2L1CB-LASSO/3_MSECB-ML1CB_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-2.log).