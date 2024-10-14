import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transformsv2
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import sys
import os

import model

from helpers import makedir
import find_nearest as find_nearest
from metrics import bb_intersection_over_union, overlap_percentage
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from plot_functions import prototypes_per_class, prototypes_per_class_v2, save_tsne, \
                                save_prototype_original_img_with_bbox, save_img_w_multiple_bbox
from dataset import MyDatasetAnalysis, ISIC_2020_split_train_val, get_WRsampler, get_df_ISIC, \
                                    ISIC_split_train_val_test, get_df_vindrmammo, VinDrMammo_split_train_val_test, \
                                    get_kvasir_df, get_WRsampler_dataset, create_polyp_datasets

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='exp/analysis/Polyps/BRAIxProtoPNet/') # PC
    parser.add_argument('--image_folder', type=str, default='C:/Users/piete/Documents/MScThesisLocal/GI_dataset/Colorectal_Polyps/Polyp_DATASET_Portal_CZE_MUMC_clean_small_v2') # PC
    #parser.add_argument('--image_folder', type=str, default='C:/Users/PCpieter/Documents/vscode/mscthesis/Polyp/Polyp_DATASET_Portal_CZE_MUMC_clean_small_v2') # PC
    parser.add_argument('--csv_path', type=str, default='finding_annotations_bbox_adjusted.csv') # PC
    parser.add_argument('--model_dir', type=str, default='saved_models/Polyp/resnet18/5fCV/') # PC
    parser.add_argument('--img_height', type=int, default=768)
    parser.add_argument('--img_width', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model', type=str, default='net_trained_last_24_8')
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--start_epoch_number', type=int, default=60)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--activation_percentile', type=int, default=94.0)
    parser.add_argument('--base_architecture', type=str, default='resnet18')
    parser.add_argument('--prototype_shape', type=tuple, default=(20, 128, 1, 1))
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')

    args = parser.parse_args()

    args.img_size = args.img_height, args.img_width

    return args

def load_model(net, path):
    checkpoint = torch.load(path, map_location='cuda:0')
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def main(args):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    train_transform = transformsv2.Compose([
    transformsv2.Resize(size=args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    ])
    
    val_transform = transformsv2.Compose([
    transformsv2.Resize(size=args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    ])

    push_transform = transformsv2.Compose([
    transformsv2.Resize(size=args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),    
    ])

    train_dataset, val_dataset, test_dataset, train_push_dataset = create_polyp_datasets(args.image_folder, train_transform, val_transform, push_transform, k=args.kfold)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # MODEL
    load_model_path = os.path.join(args.model_dir, args.model)
    
    # Assuming `ppnet` is your model, initialize it here
    ppnet = model.construct_PPNet(base_architecture=args.base_architecture,
                                pretrained=True, img_size=args.img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.num_classes,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type)

    ppnet = torch.nn.DataParallel(ppnet)
    ppnet = load_model(ppnet, load_model_path)
    device = torch.device('cuda')
    ppnet = ppnet.to(device)

    # SAVE FOLDER
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print("Save path: " + args.save_path)

    root_dir_for_saving_train_images = os.path.join(args.save_path, args.model.split('.pth')[0] + '_nearest_train')
    root_dir_for_saving_val_images = os.path.join(args.save_path, args.model.split('.pth')[0] + '_nearest_val')

    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_val_images)
    
    # SAVE PROTOTYPES IN ORIGINAL IMAGES
    load_img_dir = os.path.join(args.model_dir, 'img_24_8')
    prototype_img_folder = os.path.join(args.model_dir, 'img_24_8', 'epoch-'+str(args.start_epoch_number))
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
        print('WARNING: Not all prototypes connect most strongly to their respective classes.')\
        
    # print the prototype indexes which are wrong, prototype 0 to 9 get prefixed with B, 10 to 19 with PM
    wrong_prototypes = []
    for i, prototype in enumerate(prototype_max_connection):
        if prototype != prototype_img_identity[i]:
            if prototype == 0:
                wrong_prototypes.append(f'prototype {i} is wrongly connected to class B')
            else:
                wrong_prototypes.append(f'prototype {i} is wrongly connected to class PM')
    print('Wrong prototypes: ', wrong_prototypes)

    # #1. PLOT T-SNE WITH TOP-K ACTIVATED IMAGES BY PROTOTYPE WITH CLASS LEGEND
    # print('Saving nearest images to prototypes...')
    # for j in range(ppnet.module.num_prototypes):
    #     if j == 16:
    #         print('prototype 16')
    #     makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
    #     makedir(os.path.join(root_dir_for_saving_val_images, str(j)))
    #     save_prototype_original_img_with_bbox(load_img_dir, fname=os.path.join(root_dir_for_saving_train_images, str(j),
    #                                                             'prototype_in_original_pimg.png'),
    #                                         epoch=args.start_epoch_number,
    #                                         index=j,
    #                                         bbox_height_start=prototype_info[j][1],
    #                                         bbox_height_end=prototype_info[j][2],
    #                                         bbox_width_start=prototype_info[j][3],
    #                                         bbox_width_end=prototype_info[j][4],
    #                                         color=(0, 255, 255))

    #     save_prototype_original_img_with_bbox(load_img_dir, fname=os.path.join(root_dir_for_saving_val_images, str(j),
    #                                                             'prototype_in_original_pimg.png'),
    #                                         epoch=args.start_epoch_number,
    #                                         index=j,
    #                                         bbox_height_start=prototype_info[j][1], #ymin
    #                                         bbox_height_end=prototype_info[j][2],   #ymax
    #                                         bbox_width_start=prototype_info[j][3],  #xmin
    #                                         bbox_width_end=prototype_info[j][4],    #xmax
    #                                         color=(0, 255, 255))

    # train_prototype_labels = find_nearest.find_k_nearest_patches_to_prototypes(
    #         dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
    #         prototype_network_parallel=ppnet, # pytorch network with prototype_vectors
    #         k=args.topk,
    #         preprocess_input_function=preprocess_input_function, # normalize if needed
    #         full_save=True,
    #         root_dir_for_saving_images=root_dir_for_saving_train_images,
    #         log=print)

    # val_prototype_labels = find_nearest.find_k_nearest_patches_to_prototypes(
    #         dataloader=val_loader, # pytorch dataloader (must be unnormalized in [0,1])
    #         prototype_network_parallel=ppnet, # pytorch network with prototype_vectors
    #         k=args.topk,
    #         preprocess_input_function=preprocess_input_function, # normalize if needed
    #         full_save=True,
    #         root_dir_for_saving_images=root_dir_for_saving_val_images,
    #         log=print)

    # # 2. PLOT T-SNE WITH PROTOTYPE SEPARATION BY BIRADS CATEGORY
    # print('Saving prototype t-SNE plot...')
    # prototypes = ppnet.module.prototype_vectors.detach().cpu().numpy()
    # prototypes_per_class_v2(args.save_path, prototypes, prototype_img_identity, args.num_classes, prototype_img_folder, img_size=(100, 100), random_state=42)

if __name__ == '__main__':

    args = parse_args()
    main(args)