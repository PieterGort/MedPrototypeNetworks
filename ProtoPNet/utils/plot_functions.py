import os
import pandas as pd                                              
import seaborn as sns
import scienceplots
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from matplotlib.patches import Patch
import numpy as np
import matplotlib.patches as patches
import cv2
from matplotlib.colors import to_rgba

def save_tsne(embeddings, labels, save_path, perplexity=5, n_iter=1000, random_state=None):
    """
    Saves the t-SNE plot of the given embeddings and labels.
    """
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(embeddings)
    kl_div = tsne.kl_divergence_

    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, ax=ax, s=200, palette='viridis', alpha=0.8, edgecolor='w', linewidth=0.5)
    lim = (tsne_results.min() - 20, tsne_results.max() + 20)
    ax.set_title(f"T-SNE of Prototype Vectors (KL-divergence: {kl_div:.2f})", fontsize=20)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=15, title='Classes', title_fontsize='13')
    ax.set_xlabel('T-SNE Component 1', fontsize=20)
    ax.set_ylabel('T-SNE Component 2', fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + 'prototype_tsne_no_figs.eps', bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.savefig(save_path + 'prototype_tsne_no_figs.png', bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()


def prototypes_per_class(save_path, prototypes, prototype_labels, num_classes, prototype_img_folder, img_size=(100, 100), random_state=42):
    """
    Visualize t-SNE of prototype embeddings with images.

    Args:
        save_path (str): Path to save the resulting plot.
        prototypes (numpy.array): Array of shape (25, 128, 1, 1) representing the prototypes.
        prototype_labels (list): List of labels corresponding to each prototype.
        num_classes (int): Number of classes.
        prototype_img_folder (str): Folder containing prototype images.
        img_size (tuple): Size to which the prototype images will be resized.
        random_state (int): Random state for reproducibility.
    """
    # Reshape prototypes and run t-SNE
    prototypes = prototypes.reshape(prototypes.shape[0], -1)
    tsne = TSNE(n_components=2, perplexity=5, learning_rate='auto', n_iter=1000, random_state=random_state)
    tsne_results = tsne.fit_transform(prototypes)
    kl_div = tsne.kl_divergence_

    # Define colors for each class
    colors = sns.color_palette('bright', num_classes)

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[colors[label] for label in prototype_labels], cmap='viridis', alpha=0.6, edgecolor='k', zorder=3, s=200)

    # Make classes based on amount num_classes
    classes = [f'Class {i}' for i in range(num_classes)]
    #classes = ['benign', 'pre-malignant']

    # Add images from prototype_img_folder to each point
    for i, (x, y) in enumerate(tsne_results):
        img = Image.open(os.path.join(prototype_img_folder, f'prototype-img{i}.png'))
        img.thumbnail(img_size)
        imagebox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=True, bboxprops=dict(edgecolor=colors[prototype_labels[i]], linewidth=5), fontsize=20)
        ax.add_artist(ab)

    # Create custom legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=classes[i]) for i in range(len(classes))]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=24)

    ax.set_xlabel('T-SNE 1', fontsize=24)
    ax.set_ylabel('T-SNE 2', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-60, 60])
    plt.ylim([-120, 120])
    plt.tight_layout()
    plt.savefig(save_path + 'prototype_tsne_w_figs.eps', bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.savefig(save_path + 'prototype_tsne_w_figs.png', bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()

def prototypes_per_class_v2(save_path, prototypes, prototype_labels, num_classes, prototype_img_folder, img_size=(50, 50), random_state=42):
    """
    Visualize t-SNE of prototype embeddings with images.

    Args:
        save_path (str): Path to save the resulting plot.
        prototypes (numpy.array): Array of shape (25, 128, 1, 1) representing the prototypes.
        prototype_labels (list): List of labels corresponding to each prototype.
        num_classes (int): Number of classes.
        prototype_img_folder (str): Folder containing prototype images.
        img_size (tuple): Size to which the prototype images will be resized.
        random_state (int): Random state for reproducibility.
    """
    # Reshape prototypes and run t-SNE
    prototypes = prototypes.reshape(prototypes.shape[0], -1)
    tsne = TSNE(n_components=2, perplexity=5, learning_rate='auto', n_iter=1000, random_state=random_state)
    tsne_results = tsne.fit_transform(prototypes)
    kl_div = tsne.kl_divergence_

    # Define colors for each class
    colors = sns.color_palette('tab10', 4)
    colors = colors[num_classes:]

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[colors[label] for label in prototype_labels], alpha=0.6, edgecolor='k', zorder=3, s=300)
    
    # Make classes based on amount num_classes
    #classes = [f'Class {i}' for i in range(num_classes)]
    classes = ['Benign', 'Pre-malignant']
    
    # Add images from prototype_img_folder to each point in a circular pattern
    radius = 30
    for i, (x, y) in enumerate(tsne_results):
        img = Image.open(os.path.join(prototype_img_folder, f'prototype-img{i}.png'))
        img.thumbnail(img_size)
        imagebox = OffsetImage(img, zoom=1)
        
        angle = 2 * np.pi * (i % 12) / 12
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        new_pos = (x + dx, y + dy)
        
        ab = AnnotationBbox(imagebox, new_pos, frameon=True, bboxprops=dict(edgecolor=colors[prototype_labels[i]], linewidth=5), fontsize=20, zorder=2)
        ax.add_artist(ab)
        ax.annotate('', (x, y), xytext=new_pos, arrowprops=dict(arrowstyle='-', linestyle='dashed', color=colors[prototype_labels[i]], linewidth=1, zorder=1))

    # Create custom legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=classes[i]) for i in range(len(classes))]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=24)

    ax.set_xlabel('T-SNE 1', fontsize=24)
    ax.set_ylabel('T-SNE 2', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-60, 60])
    plt.ylim([-120, 120])
    plt.tight_layout()
    plt.savefig(save_path + 'prototype_tsne_w_figs_v2.eps', bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.savefig(save_path + 'prototype_tsne_w_figs_v2.png', bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()

def draw_and_save_combined_bbox(prototype_img_w_bbox_path, physician_bbox, save_path):
    # Load the image using PIL and convert to RGB format if necessary
    img_with_bbox = Image.open(prototype_img_w_bbox_path).convert('RGB')
    img_with_bbox = np.array(img_with_bbox)

    # Create a figure and axis to plot on
    fig, ax = plt.subplots(figsize=(20, 20))

    # Display the image
    ax.imshow(img_with_bbox)

    # Draw the physician's bounding box
    xmin, ymin, xmax, ymax = map(int, physician_bbox) # left bound, top bound, right bound, bottom bound
    width, height = (xmax - xmin), (ymax - ymin)
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=4, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add legend for the bounding boxes
    prototype_patch = patches.Patch(color='yellow', label='Prototype')
    physician_patch = patches.Patch(color='red', label='Physician')

    ax.legend(handles=[prototype_patch, physician_patch], fontsize=20)

    # Remove axis
    ax.axis('off')

    # Save the combined image with bounding boxes
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                        bbox_height_start, bbox_height_end,
                                        bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

def draw_and_save_two_bboxes(image_path, bbox1, bbox2, save_path):
    """
    Draws two bounding boxes on an image and saves the result.

    Args:
    - image_path (str): Path to the image without bounding boxes.
    - bbox1 (list): First bounding box in the format [xmin, ymin, xmax, ymax].
    - bbox2 (list): Second bounding box in the format [xmin, ymin, xmax, ymax].
    - save_path (str): Path where the combined image with bounding boxes will be saved.
    """
    # Load the image using PIL and convert to RGB format if necessary
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    # Create a figure and axis to plot on
    fig, ax = plt.subplots(figsize=(20, 20))

    # Display the image
    ax.imshow(img)

    # Draw the first bounding box
    xmin1, ymin1, xmax1, ymax1 = map(int, bbox1) # left bound, top bound, right bound, bottom bound
    width1, height1 = (xmax1 - xmin1), (ymax1 - ymin1)
    rect1 = patches.Rectangle((xmin1, ymin1), width1, height1, linewidth=6, edgecolor='red', facecolor='none')
    ax.add_patch(rect1)

    # Draw the second bounding box
    xmin2, ymin2, xmax2, ymax2 = map(int, bbox2) # left bound, top bound, right bound, bottom bound
    width2, height2 = (xmax2 - xmin2), (ymax2 - ymin2)
    rect2 = patches.Rectangle((xmin2, ymin2), width2, height2, linewidth=6, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect2)

    # Add legend for the bounding boxes
    bbox1_patch = patches.Patch(color='red', label='High activation patch')
    bbox2_patch = patches.Patch(color='cyan', label='Physician')

    #ax.legend(handles=[bbox1_patch, bbox2_patch], fontsize=30)

    # Remove axis
    ax.axis('off')

    # Save the combined image with bounding boxes
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_img_w_multiple_bbox(img_path, physician_bboxes, prototype_box, save_path):
    
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img)

    current_height, current_width = img.shape[:2]
    x_scale = current_width
    y_scale = current_height

    for bbox in physician_bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * x_scale)
        xmax = int(xmax * x_scale)
        ymin = int(ymin * y_scale)
        ymax = int(ymax * y_scale)
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
    
    xmin, ymin, xmax, ymax = prototype_box
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    width = xmax - xmin
    height = ymax - ymin
    rect_prototype = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect_prototype)

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def create_bar_chart(data, parameter_settings, metrics, title, ylabel, legend_loc='lower right', fontsize=14):
    """
    Creates a bar chart with multiple groups and different metrics for each group.

    Parameters:
    - data: 2D list or array containing the data values (each row corresponds to a metric, and each column corresponds to a parameter setting).
    - parameter_settings: List of strings representing the names of the parameter settings.
    - metrics: List of strings representing the names of the metrics.
    - title: String for the chart's title.
    - ylabel: String for the y-axis label.
    - legend_loc: Location of the legend in the plot.
    """
    n_groups = len(parameter_settings)
    n_metrics = len(metrics)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10,6))

    # Calculate the width of each bar and positions
    bar_width = 2 / n_metrics
    index = np.arange(n_groups)*3
    opacity = 1
    
    # Plotting each metric
    for i in range(n_metrics):
        plt.bar(index + i * bar_width, data[i], bar_width, alpha=opacity, label=metrics[i])


    plt.xlabel('')
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xticks(index + bar_width * (n_metrics - 1) / 2, parameter_settings, fontsize=fontsize)
    plt.legend(loc=legend_loc, fontsize=fontsize)

    # Add horizontal dashed lines at intervals of 0.05
    ymin = 0.5
    ymax = 1.0
    # ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)  # Set the y-axis limits to include the top dashed line fully
    lines = np.arange(ymin, ymax + 0.05, 0.05)  # Define lines up to a little beyond the max
    for line in lines:
        ax.hlines(line, xmin=index[0] - bar_width, xmax=index[-1] + (n_metrics - 0.5) * bar_width, colors='gray', linestyles='dashed', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def read_iou_file(file_path):
    """
    Reads the IoU report from a file and extracts activation percentiles, mean IoU, and standard deviation.
    
    Parameters:
    file_path (str): The path to the .txt file containing the IoU report.
    
    Returns:
    tuple: Three lists containing activation percentiles, mean IoU, and standard deviation.
    """
    activation_percentiles = []
    mean_IoU = []
    std_IoU = []
    
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        
        # Read the rest of the lines
        for line in file:
            parts = line.strip().split(',')
            activation_percentiles.append(float(parts[0]))
            mean_IoU.append(float(parts[1]))
            std_IoU.append(float(parts[2]))
    
    return activation_percentiles, mean_IoU, std_IoU

def plot_iou_vs_activation(activation_percentiles, mean_IoU, std_IoU):

    """
    Plots the Mean IoU vs. Activation Percentile with Standard Deviation.

    Parameters:
    activation_percentiles (list): List of activation percentile values.
    mean_IoU (list): List of mean IoU values corresponding to the activation percentiles.
    std_IoU (list): List of standard deviation values for each mean IoU.
    """
    # Set the style and context for seaborn to make the plot more visually appealing
    sns.set(style="whitegrid", context="talk")

    # Create a new figure with a specific size
    plt.figure(figsize=(8, 6))

    # Use seaborn lineplot with confidence intervals
    sns.lineplot(x=activation_percentiles, y=mean_IoU, errorbar='sd', label='Mean IoU', color='blue')

    # Adding shaded area for standard deviation
    plt.fill_between(activation_percentiles, 
                     [m - s for m, s in zip(mean_IoU, std_IoU)], 
                     [m + s for m, s in zip(mean_IoU, std_IoU)], 
                     color='blue', alpha=0.2)

    # Adding labels and title
    plt.xlabel('Activation Percentile', fontsize=14)
    plt.ylabel('Mean IoU', fontsize=14)
    plt.title('Mean IoU vs. Activation Percentile', fontsize=16)

    # Display legend
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()

def combine_iou_folds_and_save_csv(save_path, top_k_values=[1, 3, 10], num_folds=5):
    data = []
    
    # First, combine IoU fold results
    for k in top_k_values:
        combined_iou = {'Train IoU': [], 'Val IoU': []}

        for fold in range(num_folds):
            filename = os.path.join(save_path, f'IoU_report_top{k}kfold{fold}.txt')
            
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found. Skipping.")
                continue
            
            with open(filename, 'r') as f:
                current_set = None
                for line in f:
                    line = line.strip()
                    if line == 'Train IoU':
                        current_set = 'Train IoU'
                    elif line == 'Val IoU':
                        current_set = 'Val IoU'
                    elif line:
                        try:
                            iou = float(line)
                            combined_iou[current_set].append(iou)
                        except ValueError:
                            print(f"Warning: Unable to convert '{line}' to float. Skipping.")

        # Write combined results to a new file
        combined_filename = os.path.join(save_path, f'IoU_report_top{k}.txt')
        with open(combined_filename, 'w') as f:
            f.write('Train IoU\n')
            for item in combined_iou['Train IoU']:
                f.write(f'{item}\n')
            f.write('Val IoU\n')
            for item in combined_iou['Val IoU']:
                f.write(f'{item}\n')
        print(f"Combined IoU fold results saved to {combined_filename}")
    
    # Now, load combined files and create a DataFrame
    for k in top_k_values:
        filename = os.path.join(save_path, f'IoU_report_top{k}.txt')
        
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Skipping.")
            continue

        with open(filename, 'r') as f:
            current_set = None
            for line in f:
                line = line.strip()
                if line == 'Train IoU':
                    current_set = 'Training'
                elif line == 'Val IoU':
                    current_set = 'Validation'
                elif line:
                    try:
                        iou = float(line)
                        data.append({'IoU': iou, 'Set': current_set, 'Top-k': f'Top-{k}'})
                    except ValueError:
                        print(f"Warning: Unable to convert '{line}' to float. Skipping.")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_filename = os.path.join(save_path, 'iou_results.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Combined IoU results saved to {csv_filename}")
    
    # Output statistics for Top-1 Validation IoU
    print("Mean of validation IoU for top-1: ", df[(df['Top-k'] == 'Top-1') & (df['Set'] == 'Validation')]['IoU'].mean())
    print("Standard deviation of validation IoU for top-1: ", df[(df['Top-k'] == 'Top-1') & (df['Set'] == 'Validation')]['IoU'].std())

def combine_folds_coverage_and_save_csv(save_path, top_k_values=[1, 3, 10], num_folds=5):
    data = []
    
    # First, combine coverage results for each top-k value
    for k in top_k_values:
        combined_coverage = {'Train coverage': [], 'Val coverage': []}
        
        for fold in range(num_folds):
            filename = os.path.join(save_path, f'Coverage_report_top{k}kfold{fold}.txt')
            
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found. Skipping.")
                continue
            
            with open(filename, 'r') as f:
                current_set = None
                for line in f:
                    line = line.strip()
                    if line == 'Train coverage':
                        current_set = 'Train coverage'
                    elif line == 'Val coverage':
                        current_set = 'Val coverage'
                    elif line:
                        try:
                            coverage = float(line)
                            combined_coverage[current_set].append(coverage)
                        except ValueError:
                            print(f"Warning: Unable to convert '{line}' to float. Skipping.")
        
        # Write combined results to a new file
        combined_filename = os.path.join(save_path, f'Coverage_report_top{k}.txt')
        with open(combined_filename, 'w') as f:
            f.write('Train coverage\n')
            for item in combined_coverage['Train coverage']:
                f.write(f'{item}\n')
            f.write('Val coverage\n')
            for item in combined_coverage['Val coverage']:
                f.write(f'{item}\n')
        print(f"Combined fold coverage results saved to {combined_filename}")

    # Now, load combined files and create a DataFrame
    for k in top_k_values:
        filename = os.path.join(save_path, f'Coverage_report_top{k}.txt')
        
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Skipping.")
            continue

        with open(filename, 'r') as f:
            current_set = None
            for line in f:
                line = line.strip()
                if line == 'Train coverage':
                    current_set = 'Training'
                elif line == 'Val coverage':
                    current_set = 'Validation'
                elif line:
                    try:
                        coverage = float(line)
                        data.append({'coverage': coverage, 'Set': current_set, 'Top-k': f'Top-{k}'})
                    except ValueError:
                        print(f"Warning: Unable to convert '{line}' to float. Skipping.")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_filename = os.path.join(save_path, 'coverage_results.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Combined coverage results saved to {csv_filename}")

    # Output statistics for Top-1 Validation coverage
    mean_val_coverage_top1 = df[(df['Top-k'] == 'Top-1') & (df['Set'] == 'Validation')]['coverage'].mean()
    std_val_coverage_top1 = df[(df['Top-k'] == 'Top-1') & (df['Set'] == 'Validation')]['coverage'].std()
    
    print(f"Mean of validation coverage for Top-1: {mean_val_coverage_top1}")
    print(f"Standard deviation of validation coverage for Top-1: {std_val_coverage_top1}")


def create_iou_boxplot(save_path, df):
    top_k_values = [1, 3, 10]
    
    # Prepare the figure and axes
    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots(figsize=(3, 2))

    positions = np.arange(len(top_k_values))
    width = 0.35

    colors = ['#3498db', '#e74c3c']  # Blue for Training, Red for Validation
    
    for i, (set_name, color) in enumerate(zip(['Training', 'Validation'], colors)):
        data = [df[(df['Top-k'] == f'Top-{k}') & (df['Set'] == set_name)]['IoU'] for k in top_k_values]
        bp = ax.boxplot(data, positions=positions + i*width - width/2, widths=width,
                        patch_artist=True, showfliers=False, showmeans=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        meanprops={'marker': 'D', 'markerfacecolor': 'white', 
                                   'markeredgecolor': 'black', 'markersize': 4})
        
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], color=color, alpha=0.7)
        
        plt.setp(bp['medians'], color='black')
        
        for patch in bp['boxes']:
            patch.set_facecolor(to_rgba(color, 0.3))
            patch.set_edgecolor(color)

    # Set labels and ticks
    ax.set_xlabel('Top-k')
    ax.set_ylabel('IoU')
    ax.set_ylim([0, 1])
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Top-{k}' for k in top_k_values])

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.set_axisbelow(True)  # Place grid behind other elements

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1,fc=to_rgba(colors[0], 0.3), ec=colors[0], label='Training'),
                       plt.Rectangle((0,0),1,1,fc=to_rgba(colors[1], 0.3), ec=colors[1], label='Validation')]

    # Add legend on top
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.55, 1.05),
               ncol=2, frameon=True, fancybox=True)

    plt.tight_layout()
    
    # Save the plots
    png_path = os.path.join(save_path, 'IoU_boxplot_ieee.png')
    eps_path = os.path.join(save_path, 'IoU_boxplot_ieee.eps')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')

    plt.close()

    print(f"Boxplot saved as 'IoU_boxplot.png' and 'IoU_boxplot.pdf' in {save_path}")

def create_coverage_boxplot(save_path, df):

    top_k_values = [1, 3, 10]
    # Prepare the figure and axes
    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots(figsize=(3, 2))

    positions = np.arange(len(top_k_values))
    width = 0.35

    # colors = ['#3498db', '#e74c3c']  # Blue for Training, Red for Validation
    colors = ['#009e73', '#ff7f0e']  # Orange for Training, Deep Pink for Validation
    
    for i, (set_name, color) in enumerate(zip(['Training', 'Validation'], colors)):
        data = [df[(df['Top-k'] == f'Top-{k}') & (df['Set'] == set_name)]['coverage'] for k in top_k_values]
        bp = ax.boxplot(data, positions=positions + i*width - width/2, widths=width,
                        patch_artist=True, showfliers=False, showmeans=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        meanprops={'marker': 'D', 'markerfacecolor': 'white', 
                                   'markeredgecolor': 'black', 'markersize': 4})
        
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], color=color, alpha=0.7)
        
        plt.setp(bp['medians'], color='black')
        
        for patch in bp['boxes']:
            patch.set_facecolor(to_rgba(color, 0.3))
            patch.set_edgecolor(color)

    # Set labels and ticks
    ax.set_xlabel('Top-k')
    ax.set_ylabel('Coverage')
    ax.set_ylim([0, 1])
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Top-{k}' for k in top_k_values])

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.set_axisbelow(True)  # Place grid behind other elements

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1,fc=to_rgba(colors[0], 0.3), ec=colors[0], label='Training'),
                       plt.Rectangle((0,0),1,1,fc=to_rgba(colors[1], 0.3), ec=colors[1], label='Validation')]

    # Add legend on top
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.55, 1.05),
               ncol=2, frameon=True, fancybox=True)

    plt.tight_layout()
    
    # Save the plots
    png_path = os.path.join(save_path, 'Coverage_boxplot_ieee.png')
    eps_path = os.path.join(save_path, 'Coverage_boxplot_ieee.eps')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')

    plt.close()

    print(f"Boxplot saved as 'Coverage_boxplot.png' and 'Coverage_boxplot.eps' in {save_path}")

def main():
    # Use the SciencePlots styles
    plt.style.use(['science', 'ieee'])

    # Set global font sizes and line thickness for readability
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'lines.linewidth': 2,  # Increased default line thickness
        'axes.linewidth': 0.5,  # Thinner axes lines
        'grid.linewidth': 0.5,  # Thinner grid lines
    })

    save_path = 'exp/analysis/VinDrMammo/Experiment_results/44push0.8024'
    df_IoU = pd.read_csv(os.path.join(save_path, 'iou_results.csv'))
    df_coverage = pd.read_csv(os.path.join(save_path, 'coverage_results.csv'))

    # IoU
    create_iou_boxplot(save_path, df_IoU)

    # # Coverage
    create_coverage_boxplot(save_path, df_coverage)


if __name__ == '__main__':
    main()

