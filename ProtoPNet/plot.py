import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots
import os

def create_iou_boxplot(save_path, df):
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.set_palette("Set1")

    # Reduce the width of the boxes
    ax = sns.boxplot(x='Top-k', y='IoU', hue='Set', data=df, width=0.6, medianprops=dict(linewidth=3), showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'))

    plt.xlabel('')
    plt.ylabel('IoU')
    plt.ylim([0, 1])
    plt.xticks()
    plt.yticks()
    plt.legend(loc=(0.575, 0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'IoU_boxplot.png'))
    plt.savefig(os.path.join(save_path, 'IoU_boxplot.eps'), format='eps')
    plt.close()

    print(f"Boxplot saved as 'IoU_boxplot.png' in {save_path}")

def create_coverage_boxplot(save_path, df):
    print("mean of validation coverage for top-1: ", df[(df['Top-k'] == 'Top-1') & (df['Set'] == 'Validation')]['coverage'].mean())

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    ax = sns.boxplot(x='Top-k', y='coverage', hue='Set', data=df, width=0.6, medianprops=dict(linewidth=3), showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'))

    plt.xlabel('')  # Remove "Top-k" label
    plt.ylabel('Coverage')
    plt.ylim([0, 1])
    plt.xticks()
    plt.yticks()
    plt.legend(loc=(0.575, 0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Coverage_boxplot.png'))
    plt.savefig(os.path.join(save_path, 'Coverage_boxplot.eps'))
    plt.close()
    print(f"Boxplot saved as 'Coverage_boxplot.png' in {save_path}")

def main():
    # Use the SciencePlots styles
    plt.style.use(['science', 'ieee'])

    # Set global font sizes and line thickness for readability
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 10,
        'lines.linewidth': 2,  # Increased default line thickness
        'axes.linewidth': 0.5,  # Thinner axes lines
        'grid.linewidth': 0.5,  # Thinner grid lines
    })
    #save_path = 'exp/analysis/VinDrMammo/Experiment_results/44push0.8024'
    df_IoU = pd.read_csv('iou_results.csv')
    df_coverage = pd.read_csv('coverage_results.csv')

    # IoU
    create_iou_boxplot('plots', df_IoU)

    # # Coverage
    create_coverage_boxplot('plots', df_coverage)


if __name__ == '__main__':
    main()
