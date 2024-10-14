import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import cv2
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Subset

from utils.preprocess import undo_preprocess_input_function

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def make_one_hot_sklearn(labels, n_classes):
    encoder = OneHotEncoder(categories=[range(n_classes)], sparse_output=False)
    return encoder.fit_transform(labels.reshape(-1, 1))

def makedir(path):
    '''S
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    # threshold = percentile*activation_map.max()/100
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def save_preprocessed_img(fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
        undo_preprocessed_img = undo_preprocess_input_function(img_copy)
        print('image index {0} in batch'.format(index))
        undo_preprocessed_img = undo_preprocessed_img[0]
        undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
        undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
        
        plt.imsave(fname, undo_preprocessed_img)
        return undo_preprocessed_img

def save_prototype(fname, epoch, index, load_img_dir):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)
    
def save_prototype_self_activation(fname, epoch, index, load_img_dir):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)

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

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                    bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)
    return img_rgb_float

def combined_images(original_img, 
                    overlayed_img, 
                    patch_in_img, 
                    high_act_patch, 
                    idx_hight_act,
                    prototype_img_identity, 
                    predicted_cls, 
                    actual_label, 
                    activation_map_threshold, 
                    save_path):
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.4])  # Adjust the last ratio for smaller width

    # Creating each subplot in the figure using the gridspec module
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original_img)
    ax0.axis('off')  # Ensures that the axis is turned off
    ax0.set_title('Original Image', fontsize=10)

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(overlayed_img)
    ax1.axis('off')  # Ensures that the axis is turned off
    ax1.set_title('Prototype Self-Activation Map', fontsize=10)

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(patch_in_img)
    ax2.axis('off')  # Ensures that the axis is turned off
    ax2.set_title('Most Highly Activated Patch in Original Image', fontsize=10)

    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(high_act_patch)
    ax3.axis('off')  # Ensures that the axis is turned off
    ax3.set_title('Most Highly Activated Patch by Prototype (class): ', fontsize=10)

    plt.suptitle(f'Predicted class: {predicted_cls}, Actual class: {actual_label}, Activation map threshold: {activation_map_threshold} %, prototype number: {idx_hight_act}, prototype class: {prototype_img_identity[idx_hight_act]}', fontsize=12, x=0.5, y=0.08)
    plt.savefig(os.path.join(save_path, 'combined_images.png'))

def create_small_dataloader(dataset, batch_size, num_workers=0, sampler=None, shuffle=False, pin_memory=True, subset_size=100):
    """ Create a DataLoader with only a subset of the dataset. """
    # Select a random subset of indices for the dataset
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset = Subset(dataset, indices)
    
    # If a sampler is used, modify it to work only with the subset of indices
    if sampler is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        shuffle = False  # Disable shuffling if using a sampler
    
    # Create the DataLoader with the subset
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers,
                        sampler=sampler, shuffle=shuffle, pin_memory=pin_memory)
    return loader

