import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys

import re

import os

from utils.helpers import makedir
import models.model
import utils.find_nearest as find_nearest
import train_and_test as tnt
from data.dataset import ISICDataset2, get_df, split_train_val_test, get_WRsampler

from utils.preprocess import preprocess_input_function

import argparse


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    image_folder = args.image_folder
    labels_file = args.labels_file
    load_model_dir = args.modeldir
    load_model_name = args.model
    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    # settings
    from settings import num_workers_snellius, num_workers_local

    # load the model
    print('load model from ' + load_model_path)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size

    # # load the data
    # # must use unaugmented (original) dataset
    # from settings import train_push_dir, test_dir
    # train_dir = train_push_dir

    image_folder = '/gpfs/work5/0/prjs0976/data'
    try:
        os.listdir(image_folder)
    except FileNotFoundError:
        image_folder = 'C:/Users/piete/Documents/MScThesisLocal/data'
        try:
            os.listdir(image_folder)
        except FileNotFoundError:
            image_folder = 'C:/Users/PCpieter/Documents/vscode/mscthesis/data'

    print('Image folder is:', image_folder)

    # print('Labels file is:', labels_file)

    if image_folder.startswith('/gpfs/work5'):
        num_workers = num_workers_snellius
    else:
        num_workers = num_workers_local

    print("num_workers is:", num_workers)

    # ## TESTING THE WINNING DATASET FUNCTIONALITY
    df, df_test, mel_idx = get_df('newfold', 9, image_folder, 512, False)
    train_df, val_df, test_df = split_train_val_test(df, val_size=0.25, random_state=42, num_splits=5, k=0)

    # transforms_train, transforms_val = get_transforms(512)

    # # # split the dataset with a fixed seed (42)
    # train_df, val_df = split_dataframe(pd.read_csv(labels_file), val_size=val_size)

    # initiate WeightedRandomSampler
    WRsampler = get_WRsampler(train_df)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    

    train_dataset = ISICDataset2(train_df, transform=train_transform)
    test_dataset = ISICDataset2(val_df, transform=train_transform)

    train_batch_size = 16
    test_batch_size = 16

    train_loader = DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              num_workers=num_workers, 
                              sampler=WRsampler, 
                              shuffle=False, 
                              pin_memory=True)

    test_loader = DataLoader(test_dataset, 
                            batch_size=test_batch_size, 
                            num_workers=num_workers, 
                            shuffle=False, 
                            pin_memory=True)

    
    global_analysis_folder = "exp/global_analysis/"
    if not os.path.exists(global_analysis_folder):
        os.makedirs(global_analysis_folder)

    features_name = load_model_dir.split('/')[-2]
    experiment_name = load_model_dir.split('/')[-1]

    global_analysis_folder = os.path.join(global_analysis_folder, features_name, experiment_name)

    print("Folder where the analysis will be saved", global_analysis_folder)
    
    root_dir_for_saving_train_images = os.path.join(global_analysis_folder,
                                                    load_model_name.split('.pth')[0] + '_nearest_train')
    root_dir_for_saving_test_images = os.path.join(global_analysis_folder,
                                                    load_model_name.split('.pth')[0] + '_nearest_test')
    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)

    # save prototypes in original images
    load_img_dir = os.path.join(load_model_dir, 'img')
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'bb'+str(start_epoch_number)+'.npy'))
    def save_prototype_original_img_with_bbox(fname, epoch, index,
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

    for j in range(ppnet.num_prototypes):
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
        save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            epoch=start_epoch_number,
                                            index=j,
                                            bbox_height_start=prototype_info[j][1],
                                            bbox_height_end=prototype_info[j][2],
                                            bbox_width_start=prototype_info[j][3],
                                            bbox_width_end=prototype_info[j][4],
                                            color=(0, 255, 255))
        save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            epoch=start_epoch_number,
                                            index=j,
                                            bbox_height_start=prototype_info[j][1],
                                            bbox_height_end=prototype_info[j][2],
                                            bbox_width_start=prototype_info[j][3],
                                            bbox_width_end=prototype_info[j][4],
                                            color=(0, 255, 255))

    k = 5

    find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            k=k+1,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_train_images,
            log=print)

    find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            k=k,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_test_images,
            log=print)


if __name__ == '__main__':

    sys.path.insert(0, './models')
    gpuid = '0'

    image_folder = '/Users/PCpieter/Documents/vscode/mscthesis/data'
    csv_path = '/Users/PCpieter/Documents/vscode/mscthesis/data/combined_train.csv'

    # image_folder = '/Users/piete/Documents/MScThesisLocal/data'
    # csv_path = '/Users/piete/Documents/MScThesisLocal/data/combined_train.csv'
    modeldir = 'exp/saved_models/resnet18/base_512_#2'
    model = '40push0.8979.pth'

    # Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--image_folder', type=str, default=image_folder)
    parser.add_argument('--labels_file', type=str, default=csv_path)
    parser.add_argument('--modeldir', type=str, default=modeldir)
    parser.add_argument('--model', type=str, default=model)
    #parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
    args = parser.parse_args()

    main()
    