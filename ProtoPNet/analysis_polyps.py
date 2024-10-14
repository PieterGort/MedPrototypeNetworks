
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transformsv2
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2
from PIL import Image
import sys
import re

import os
import copy



from utils.helpers import makedir, find_high_activation_crop, save_preprocessed_img, imsave_with_bbox, combined_images
from models.model import PPNet
import models.push
import train_and_test as tnt
import utils.save
import utils.find_nearest as find_nearest
from utils.metrics import bb_intersection_over_union
from utils.log import create_logger
from utils.preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from utils.plot_functions import prototypes_per_class_v2, save_tsne
from data.dataset import MyDatasetAnalysis, ISIC_2020_split_train_val, get_WRsampler, get_df_ISIC, \
                                    ISIC_split_train_val_test, get_df_vindrmammo, VinDrMammo_split_train_val_test, \
                                    get_kvasir_df, get_WRsampler_dataset, create_polyp_datasets

import argparse
def filter_no_annotations(df):
    # Create a mask for rows where xmin, xmax, ymin, ymax are all 0
    no_annotation_mask = (df['xmin'] == 0) & (df['xmax'] == 0) & (df['ymin'] == 0) & (df['ymax'] == 0)
    # Invert the mask to keep rows with valid annotations
    df_filtered = df[~no_annotation_mask]
    return df_filtered

def main(args):

    # df = get_kvasir_df(args.image_folder)

    # train_df = df[df['split'] == 'train']
    # test_df = df[df['split'] == 'val']

    # # # DATASET
    # # df = get_df_vindrmammo(args.image_folder, args.csv_path, args.img_size, args.num_classes)
    # # train_df, val_df, test_df = VinDrMammo_split_train_val_test(df, random_state=42, num_splits=5, k=args.kfold)
    
    # # print("length of train_df: " + str(len(train_df)))
    # # print("length of val_df: " + str(len(val_df)))
    # # print("length of test_df: " + str(len(test_df)))

    # # train_df = filter_no_annotations(train_df)
    # # val_df = filter_no_annotations(val_df)
    # # test_df = filter_no_annotations(test_df)

    # # print("length of train_df: " + str(len(train_df)))
    # # print("length of val_df: " + str(len(val_df)))
    # # print("length of test_df: " + str(len(test_df)))
    
    train_transform = transformsv2.Compose([
    transformsv2.Resize(size=args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    ])
    
    test_transform = transformsv2.Compose([
    transformsv2.Resize(size=args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    ])

    # train_dataset = MyDatasetAnalysis(train_df, transform=train_transform)
    # test_dataset = MyDatasetAnalysis(test_df, transform=test_transform)

    train_dataset, val_dataset, test_dataset, _ = create_polyp_datasets(args.image_folder, train_transform, test_transform, train_transform, num_splits=5, k=0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        
    # MODEL
    load_model_path = os.path.join(args.model_dir, args.model)
    ppnet = torch.load(os.path.join(load_model_path))
    ppnet = torch.nn.DataParallel(ppnet)
    ppnet = ppnet.cuda()
    ppnet_multi = ppnet

    # SAVE FOLDER
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print("Save path: " + args.save_path)

    root_dir_for_saving_train_images = os.path.join(args.save_path, args.model.split('.pth')[0] + '_nearest_train')
    root_dir_for_saving_test_images = os.path.join(args.save_path, args.model.split('.pth')[0] + '_nearest_test')

    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)
    
    # SAVE PROTOTYPES IN ORIGINAL IMAGES
    load_img_dir = os.path.join(args.model_dir, 'img')
    prototype_img_folder = os.path.join(args.model_dir, 'img', 'epoch-'+str(args.start_epoch_number))
    prototype_info = np.load(os.path.join(prototype_img_folder, 'bb'+str(args.start_epoch_number)+'.npy'))
    prototype_img_identity = prototype_info[:, -1]
    print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    print('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.module.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.module.num_prototypes:
        print('All prototypes connect most strongly to their respective classes.')
        print('Correct prototype connections: ', np.sum(prototype_max_connection == prototype_img_identity))
    else:
        print('WARNING: Not all prototypes connect most strongly to their respective classes.')

    # 1. PLOT T-SNE WITH TOP-K ACTIVATED IMAGES BY PROTOTYPE WITH CLASS LEGEND
    print('Saving nearest images to prototypes...')
    for j in range(ppnet.module.num_prototypes):
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
        save_prototype_original_img_with_bbox(load_img_dir, fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            epoch=args.start_epoch_number,
                                            index=j,
                                            bbox_height_start=prototype_info[j][1],
                                            bbox_height_end=prototype_info[j][2],
                                            bbox_width_start=prototype_info[j][3],
                                            bbox_width_end=prototype_info[j][4],
                                            color=(0, 255, 255))
        
        save_prototype_original_img_with_bbox(load_img_dir, fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            epoch=args.start_epoch_number,
                                            index=j,
                                            bbox_height_start=prototype_info[j][1], #ymin
                                            bbox_height_end=prototype_info[j][2],   #ymax
                                            bbox_width_start=prototype_info[j][3],  #xmin
                                            bbox_width_end=prototype_info[j][4],    #xmax
                                            color=(0, 255, 255))

    find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            k=args.topk,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_train_images,
            log=print)

    find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            k=args.topk,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_test_images,
            log=print)

    # 2. PLOT T-SNE WITH PROTOTYPE SEPARATION BY BIRADS CATEGORY
    print('Saving prototype t-SNE plot...')
    prototypes = ppnet.module.prototype_vectors.detach().cpu().numpy()
    prototypes_per_class_v2(args.save_path, prototypes, prototype_img_identity, args.num_classes, prototype_img_folder, img_size=(75, 75), random_state=42)

    # Test old tsne function
    print('Saving prototype t-SNE plot with old function...')
    save_tsne(prototypes, prototype_img_identity, args.save_path, perplexity=5, n_iter=1000, random_state=42)
    
    
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

if __name__ == '__main__':
    sys.path.insert(0, './models')
    sys.path.insert(0, './exp')
    sys.path.insert(0, './utils')

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='exp/analysis/Polyps/resnet18/ProtoPNet_Params') # PC
    parser.add_argument('--image_folder', type=str, default='C:/Users/PCpieter/Documents/vscode/mscthesis/Polyp/Polyp_DATASET_Portal_CZE_MUMC_clean_small_v2') # PC
    parser.add_argument('--csv_path', type=str, default=r'..') # PC
    parser.add_argument('--model_dir', type=str, default='exp/saved_models/Polyps/resnet18/ProtoPNet_Params/') # PC
    parser.add_argument('--img_height', type=int, default=768)
    parser.add_argument('--img_width', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='final_model.pth')
    parser.add_argument('--experiment_name', type=str, default=' ')
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--start_epoch_number', type=int, default=40)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()

    args.img_size = args.img_height, args.img_width

    main(args)
