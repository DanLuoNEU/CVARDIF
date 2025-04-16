''' Visualize log items with epoch on Tensorboard
    25/01, Dan
'''
import os
import ipdb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def main():
    '''CVARDIF log vis'''
    list_paths = [
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs12.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs32.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs32_lr2-1e-3.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs32_dif-init.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs32_dif-init_lr2-1e-3.log",
        "/data/Dan/202501_SBAR/1_L1RC_desc/1_lr-milestone-try/1_1_L1RC_desc_lamf1e-2_lam2-1e+1_aC1e+0_bs32_dif-init_lr2-1e+0_ms40-50-60.log"
    ]
    # Directory to save TensorBoard logs
    log_dir = "notes_exp/SBAR_SparseCoding/0_training/tb_logs"

    # Create a SummaryWriter for each experiment
    experiment_names = [
                        # "bs12", "bs32", "bs32_lr2-1e-3",
                        # "bs32_dif-init", "bs32_dif-init_lr2-1e-3",
                        "bs32_dif-init_lr2-1e+0_ms-try"]
    writers = {name: SummaryWriter(log_dir=os.path.join(log_dir, name)) for name in experiment_names}

    epoch_data={}
    for i,path in enumerate(list_paths):
        with open(path,'r') as f: lines = f.readlines()
        epoch_data[experiment_names[i]]={}
        for line in lines:
            if not line.startswith('Test'): continue
            words = line.split()
            epoch = int(words[2])
            if epoch > 100: break
            
            MSE = float(words[10])
            L1_C = float(words[6])
            Sp = float(words[15].split(':')[-1])
            
            epoch_data[experiment_names[i]][epoch] = {'MSE_C':MSE,
                                                      'L1_C':L1_C,
                                                      'Sparsity':Sp}


    # Write data to TensorBoard logs
    for experiment, data in epoch_data.items():
        writer = writers[experiment]
        for epoch, metrics in data.items():
            for item, value in metrics.items():
                writer.add_scalar(f"{item}", value, epoch)

    # Close all writers
    for writer in writers.values():
        writer.close()

    print(f"TensorBoard logs saved in '{log_dir}'.")

def main_v1():
    '''CVARDIF log vis'''
    list_paths = [
        # "/data/Dan/202111_CVAR/202410_CVARDIF/2501_1_wiRH/0_yuexi_test_l1loss/yuexi_test_l1loss.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs32.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/1_L1RC_desc_lamf1e-2_lam2-1e+4_aC1e+3_bs32_dif-init_lr2-1e-3.log"
        ## 25/02/06, pole init 1
        # "/data/Dan/202111_CVAR/202410_CVARDIF/2501_1_wiRH/0_yuexi_test_l1loss/yuexi_test_l1loss.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/3_250206_pole_init/1_L1RC_desc_3_lamf1e-2_lam2-1e+4_aC1e+3_bs32_12q_lr2-1e-4.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/3_250206_pole_init/1_L1RC_desc_3_lamf1e-2_lam2-1e+4_aC1e+3_bs32_cvar-ft_lr2-1e-4.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/3_250206_pole_init/1_L1RC_desc_3_lamf1e-2_lam2-1e+4_aC1e+3_bs32_uni_lr2-1e-4.log"
        ## 25/02/07, pole init 2
        # "/data/Dan/202501_SBAR/1_L1RC_desc/3_250206_pole_init/1_L1RC_desc_3_lamf1e-2_lam2-1e+4_aC1e+3_bs32_cvar-init-range1-1_lr2-1e-4.log",
        # "/data/Dan/202501_SBAR/1_L1RC_desc/3_250206_pole_init/1_L1RC_desc_3_lamf1e-2_lam2-1e+4_aC1e+3_bs32_cvar-init-range3_lr2-1e-4.log"
        ## 25/02/11, realD clamp range fixed
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-1.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-2.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-3.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-4.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e+0.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e+1.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e+2.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e+3.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth502_bs32_clamp-cvar-pret_test.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-1.log",
        "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-2.log",
        "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/3_MSECB-ML1CB_0_lamf1e-2_gth505_bs32_clamp-cvar-pret_test.log",
        # "/data/Dan/202501_SBAR/3_MSECB-ML1CB_desc/0_250210_clamp_LASSO-Loss/1_sim-init/3_MSECB-ML1CB_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-2.log"
        # 250221 ML1CBcol
        "/data/Dan/202501_SBAR/5_ML2CBL1CBcol/0_250221_clamp_LASSO-Loss_o1/5_ML2CBL1CBcol_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e+0.log",
        "/data/Dan/202501_SBAR/5_ML2CBL1CBcol/0_250221_clamp_LASSO-Loss_o1/5_ML2CBL1CBcol_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-1.log",
        "/data/Dan/202501_SBAR/5_ML2CBL1CBcol/0_250221_clamp_LASSO-Loss_o1/5_ML2CBL1CBcol_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-2.log",
        "/data/Dan/202501_SBAR/5_ML2CBL1CBcol/0_250221_clamp_LASSO-Loss_o1/5_ML2CBL1CBcol_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-3.log",
        "/data/Dan/202501_SBAR/5_ML2CBL1CBcol/0_250221_clamp_LASSO-Loss_o1/5_ML2CBL1CBcol_0_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-4.log",
        "/data/Dan/202501_SBAR/5_ML2CBL1CBcol/1_250221_clamp_LASSO-Loss_o2/5_ML2CBL1CBcol_1_lamf1e-2_gth505_lam2-5e-1_a1e-2_bs32_clamp_lr2-1e-2.log"
    ]
    # Directory to save TensorBoard logs
    log_dir = "notes_exp/SBAR_SparseCoding/5_clamp-L2L1CBcol-LASSO/tb_logs"

    # Create a SummaryWriter for each experiment
    experiment_names = [
                        # "cvar_pret",
                        # "bs32_12q","bs32_cvar-ft", "bs32_uni"
                        # "cvar-init-range1_1", "cvar-init-range3"
                        # "L1RC_clamp", "L1RB_clamp"
                        # "gth502_lr1e-1","gth502_lr1e-2","gth502_lr1e-3","gth502_lr1e-4","gth502_lr1e+0","gth502_lr1e+1","gth502_lr1e+2","gth502_lr1e+3",
                        # "gth502_cvar-test","gth505_lr1e-1","gth505_lr1e-2","gth505_cvar-test"
                        "gth505_lr1e-2","gth505_cvar-test","gth505_lr1e+0_col","gth505_lr1e-1_col","gth505_lr1e-2_col","gth505_lr1e-3_col","gth505_lr1e-4_col", "gth505_lr1e-2_161*ML1"
                        ]
    writers = {name: SummaryWriter(log_dir=os.path.join(log_dir, name)) for name in experiment_names}

    epoch_data={}
    for i,path in enumerate(list_paths):
        with open(path,'r') as f: lines = f.readlines()
        epoch_data[experiment_names[i]]={}
        for line in lines:
            if not line.startswith('Test'): continue
            words = line.split()
            epoch = int(words[2])
            if epoch > 50: break
            
            L1_C = float(words[6])
            MSE = float(words[10])
            L1_C_B = float(words[8])
            L1_B = float(words[12])
            MSE_B = float(words[14])

            Sp = float(words[15].split(':')[-1])
            
            epoch_data[experiment_names[i]][epoch] = {'MSE_C':MSE,
                                                      'L1_C':L1_C,
                                                      'MSE_B':MSE_B,
                                                      'L1_C_B':L1_C_B,
                                                      'L1_B':L1_B,
                                                      'Sparsity':Sp}


    # Write data to TensorBoard logs
    for experiment, data in epoch_data.items():
        writer = writers[experiment]
        for epoch, metrics in data.items():
            for item, value in metrics.items():
                writer.add_scalar(f"{item}", value, epoch)

    # Close all writers
    for writer in writers.values():
        writer.close()

    print(f"TensorBoard logs saved in '{log_dir}'.")


def main_test():
    list_paths = [
        "/data/Dan/202501_SBAR/1_L1RC_desc/3_250206_pole_init/lamf1e-2_lam2-1e+4_aC1e+3_bs32_uni_lr2-1e-4/bs32_uni-gth.log"
    ]
    # Directory to save TensorBoard logs
    log_dir = "/home/dan/ws/202411_SBAR/2501_SBAR/notes_exp/SBAR_SparseCoding/2_pole-init-1"

    # Create a SummaryWriter for each experiment
    experiment_names = ["uni_gth"]
    # writers = {name: SummaryWriter(log_dir=os.path.join(log_dir, name)) for name in experiment_names}

    epoch_data={}
    for i,path in enumerate(list_paths):
        with open(path,'r') as f: lines = f.readlines()
        epoch_data[experiment_names[i]]={}
        # ipdb.set_trace()
        for line in lines:
            if not line.startswith('Test'): continue
            words = line.split()

            lam = float(words[2])
            L1_C = float(words[6])
            MSE = float(words[10])
            L1_C_B = float(words[8])
            L1_B = float(words[12])
            MSE_B = float(words[14])
            Sp = float(words[-2].split(':')[-1])
            
            epoch_data[experiment_names[i]][lam] = {'MSE_C':MSE,
                                                    'L1_C':L1_C,
                                                    'MSE_B':MSE_B,
                                                    'L1_C_B':L1_C_B,
                                                    'L1_B':L1_B,
                                                    'Sparsity':Sp}

    for experiment, data in epoch_data.items():
        for item in ['MSE_B', 'L1_C_B']:
            # Prepare a new figure for each metric
            fig, ax = plt.subplots()

            # Plot each metric (MSE_C, L1_C, Sparsity) against lam (float)
            for lam, metrics in data.items():
                value = metrics[item]
                ax.plot(lam, value, marker='o', label=item)  # Plot with 'lam' as float on x-axis

            # Set labels and title
            ax.set_xlabel('RH-FISTA gth')
            ax.set_ylabel(f'{item}')
            # ax.legend()

            # Save the figure as a separate file for each metric
            fig_filename = os.path.join(log_dir,f"{item} g_th.png")
            fig.savefig(fig_filename)

            # Close the figure to release memory
            plt.close(fig)


if __name__ == "__main__":
    # main()
    main_v1()
    # main_test()