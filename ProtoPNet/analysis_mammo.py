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

from utils.helpers import makedir
import utils.find_nearest as find_nearest
from utils.metrics import bb_intersection_over_union, coverage_function
from utils.preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from utils.plot_functions import prototypes_per_class, prototypes_per_class_v2, save_tsne, \
                                save_prototype_original_img_with_bbox, save_img_w_multiple_bbox
from data.dataset import MyDatasetAnalysis, ISIC_2020_split_train_val, get_WRsampler, get_df_ISIC, \
                                    ISIC_split_train_val_test, get_df_vindrmammo, VinDrMammo_split_train_val_test, \
                                    get_kvasir_df, get_WRsampler_dataset


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='exp/analysis/VinDrMammo/Classes_00111/Experiment_results/') # PC
    parser.add_argument('--image_folder', type=str, default='C:/Users/piete/Documents/MScThesisLocal/VinDrMammo/processed_images') # PC
    parser.add_argument('--csv_path', type=str, default='finding_annotations_bbox_adjusted.csv') # PC
    parser.add_argument('--model_dir', type=str, default='exp/saved_models/VinDrMammo/resnet34/Classes_00111/') # PC
    parser.add_argument('--img_height', type=int, default=1536)
    parser.add_argument('--img_width', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model', type=str, default='55push0.7708.pth')
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--start_epoch_number', type=int, default=55)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--activation_percentile', type=int, default=94.0)

    args = parser.parse_args()

    args.img_size = args.img_height, args.img_width

    return args

def main(args):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    # DATASET
    df = get_df_vindrmammo(args.image_folder, args.csv_path, args.num_classes)
    train_df, val_df, _ = VinDrMammo_split_train_val_test(df, random_state=42, num_splits=5, k=args.kfold)
    
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

    train_dataset = MyDatasetAnalysis(train_df, transform=train_transform)
    val_dataset = MyDatasetAnalysis(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        
    # MODEL
    load_model_path = os.path.join(args.model_dir, args.model)
    ppnet = torch.load(os.path.join(load_model_path))
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    # SAVE FOLDER
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print("Save path: " + args.save_path)

    root_dir_for_saving_train_images = os.path.join(args.save_path, args.model.split('.pth')[0] + '_nearest_train')
    root_dir_for_saving_val_images = os.path.join(args.save_path, args.model.split('.pth')[0] + '_nearest_val')

    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_val_images)
    
    # SAVE PROTOTYPES IN ORIGINAL IMAGES
    load_img_dir = os.path.join(args.model_dir, 'img')
    prototype_img_folder = os.path.join(args.model_dir, 'img', 'epoch-'+str(args.start_epoch_number))
    prototype_info = np.load(os.path.join(prototype_img_folder, 'bb'+str(args.start_epoch_number)+'.npy'))
    prototype_img_identity = prototype_info[:, -1]

    #1. PLOT T-SNE WITH TOP-K ACTIVATED IMAGES BY PROTOTYPE WITH CLASS LEGEND
    print('Saving nearest images to prototypes...')
    for j in range(ppnet.num_prototypes):
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_val_images, str(j)))
        save_prototype_original_img_with_bbox(load_img_dir, fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            epoch=args.start_epoch_number,
                                            index=j,
                                            bbox_height_start=prototype_info[j][1],
                                            bbox_height_end=prototype_info[j][2],
                                            bbox_width_start=prototype_info[j][3],
                                            bbox_width_end=prototype_info[j][4],
                                            color=(0, 255, 255))

        save_prototype_original_img_with_bbox(load_img_dir, fname=os.path.join(root_dir_for_saving_val_images, str(j),
                                                                'prototype_in_original_pimg.png'),
                                            epoch=args.start_epoch_number,
                                            index=j,
                                            bbox_height_start=prototype_info[j][1], #ymin
                                            bbox_height_end=prototype_info[j][2],   #ymax
                                            bbox_width_start=prototype_info[j][3],  #xmin
                                            bbox_width_end=prototype_info[j][4],    #xmax
                                            color=(0, 255, 255))

    _, _, all_train_bboxes = find_nearest.find_k_nearest_patches_to_prototypes_with_filenames(
            dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            k=args.topk,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_train_images,
            log=print,
            activation_percentile=args.activation_percentile)

    _, _, all_val_bboxes = find_nearest.find_k_nearest_patches_to_prototypes_with_filenames(
            dataloader=val_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            k=args.topk,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_val_images,
            log=print,
            activation_percentile=args.activation_percentile)

    # 2. PLOT T-SNE WITH PROTOTYPE SEPARATION BY BIRADS CATEGORY
    print('Saving prototype t-SNE plot...')
    prototypes = ppnet.prototype_vectors.detach().cpu().numpy()
    prototypes_per_class(args.save_path, 
                                      prototypes, prototype_img_identity, 
                                      args.num_classes, prototype_img_folder, 
                                      img_size=(75, 75), random_state=42)
    
    train_iou_results_list = []
    val_iou_results_list = []
    train_overlap_results_list = []
    val_overlap_results_list = []
    # 3. FIND SIMILARITY OF PROTOTYPES TO ANNOTATED BOUNDING BOX FROM THE DATASET
    print('Finding similarity of prototypes to physician annotated bounding boxes...')
    for saved_images_directory in [root_dir_for_saving_train_images, root_dir_for_saving_val_images]:

        bbox_results_iou = []
        bbox_results_overlap = []

        if 'train' in saved_images_directory:
            print('Train set...')
            print('Saved images directory:', saved_images_directory)
            current_set = 'train'
        elif 'val' in saved_images_directory:
            print('Val set...')
            print('Saved images directory:', saved_images_directory)
            current_set = 'val'
        else:
            print('Skipping unrecognized directory:', saved_images_directory)
            continue

        for prototype_id in range(ppnet.num_prototypes):
            print('Prototype ID:', prototype_id)
            prototype_dir = os.path.join(saved_images_directory, str(prototype_id))
            filenames_path = os.path.join(prototype_dir, 'filenames.npy')
            prototype_bbox_path = os.path.join(prototype_dir, 'bboxes.npy')

            if not os.path.exists(filenames_path):
                print('No filenames.npy file found in directory:', filenames_path)
                continue
            if not os.path.exists(prototype_bbox_path):
                print('No bboxes.npy file found in directory:', prototype_bbox_path)
                continue

            filenames = np.load(filenames_path)
            prototype_bboxes = np.load(prototype_bbox_path) # xmin, ymin, xmax, ymax, prototype_id, idx
            
            for idx, image_path in enumerate(filenames):
                try:
                    if not os.path.exists(image_path):
                        image_path_split = image_path.split('\\')[-2:]
                        image_path = os.path.join(args.image_folder, *image_path_split)
                        
                    if not os.path.exists(image_path):
                        raise FileNotFoundError('Image path does not exist: ' + image_path)

                    nearest_img_path = os.path.join(prototype_dir, f'nearest-{idx+1}_original.png')
                    prototype_bbox = prototype_bboxes[idx, :4]

                    image_row_df = df[df['filepath'] == image_path]
                    if image_row_df.empty:
                        print('No matching image row found in DataFrame for image path:', image_path)
                        continue

                    image_row_df = image_row_df.iloc[0]
                    physician_bboxes = image_row_df['bounding_boxes']

                    # check if the physician bounding box is valid, meaning that not all values are -1, if not proceed to the next image
                    if all([all([y < 0 for y in bb]) for bb in physician_bboxes]):
                        print('No valid physician bounding boxes found for prototype:', prototype_id, 'image number', idx+1)
                        continue

                    save_bbox_similarities_path = os.path.join(prototype_dir, f'bounding_box_similarity-{idx+1}.png')
                    save_img_w_multiple_bbox(nearest_img_path, physician_bboxes, prototype_bbox, save_bbox_similarities_path)

                    # calculate the intersection over union (IoU) between the prototype and physician bounding boxes
                    for physician_bbox in physician_bboxes:
                        xmin, ymin, xmax, ymax = physician_bbox
                        xmin, xmax = xmin * args.img_width, xmax * args.img_width
                        ymin, ymax = ymin * args.img_height, ymax * args.img_height
                        scaled_physician_bbox = [xmin, ymin, xmax, ymax]

                        # Calculate IoU
                        IoU = bb_intersection_over_union(prototype_bbox, scaled_physician_bbox)
                        bbox_results_iou.append(IoU)

                        coverage = coverage_function(scaled_physician_bbox, prototype_bbox)
                        bbox_results_overlap.append(coverage) 
                        
                        if current_set == 'train':
                            train_iou_results_list.append(IoU)
                            train_overlap_results_list.append(coverage)
                        else:
                            val_iou_results_list.append(IoU)
                            val_overlap_results_list.append(coverage)
                        
                    print(f'IoU for Prototype: {prototype_id}, Image: {idx+1}, Physician Bbox: {physician_bboxes.index(physician_bbox)+1} = {IoU}')
                    print(f'Overlap Percentage for Prototype: {prototype_id}, Image: {idx+1}, Physician Bbox: {physician_bboxes.index(physician_bbox)+1} = {coverage}')
                    
                except Exception as e:
                    print('Error processing image:', image_path, ':', str(e))

        # Calculate and log IoU statistics for the current set
        if bbox_results_iou:
            mean_iou = np.mean(bbox_results_iou)
            std_iou = np.std(bbox_results_iou)
            median_iou = np.median(bbox_results_iou)
            percentile_iou = np.percentile(bbox_results_iou, [25, 75])

            print(f'Mean IoU over {len(bbox_results_iou)} images for set {saved_images_directory}: {mean_iou}')
            print(f'Standard Deviation of IoU: {std_iou}')
            print(f'Median IoU: {median_iou}')
            print(f'25th Percentile IoU: {percentile_iou[0]}')
            print(f'75th Percentile IoU: {percentile_iou[1]}')

        # Calculate and log Overlap Percentage statistics for the current set
        if bbox_results_overlap:
            mean_overlap = np.mean(bbox_results_overlap)
            std_overlap = np.std(bbox_results_overlap)
            median_overlap = np.median(bbox_results_overlap)
            percentile_overlap = np.percentile(bbox_results_overlap, [25, 75])

            print(f'Mean Overlap Percentage over {len(bbox_results_overlap)} images: {mean_overlap}')
            print(f'Standard Deviation of Overlap: {std_overlap}')
            print(f'Median Overlap: {median_overlap}')
            print(f'25th Percentile Overlap: {percentile_overlap[0]}')
            print(f'75th Percentile Overlap: {percentile_overlap[1]}')

    print('Done!')

    return train_iou_results_list, val_iou_results_list, train_overlap_results_list, val_overlap_results_list

if __name__ == '__main__':
    args = parse_args()
    train_iou_results_list, val_iou_results_list, train_overlap_results_list, val_overlap_results_list = main(args)

    iou_report_path = 'IoU_report' + '_top' + str(args.topk) + 'kfold' + str(args.kfold) + '.txt'
    coverage_report_path = 'Coverage_report' + '_top' + str(args.topk) + 'kfold' + str(args.kfold) + '.txt'

    # Save the IoU results to a file
    with open(os.path.join(args.save_path, iou_report_path), 'w') as f:
        f.write('Train IoU\n')
        for item in train_iou_results_list:
            f.write(f'{item}\n')
        f.write('Val IoU\n')
        for item in val_iou_results_list:
            f.write(f'{item}\n')

    # Save the coverage results to a file
    with open(os.path.join(args.save_path, coverage_report_path), 'w') as f:
        f.write('Train coverage\n')
        for item in train_overlap_results_list:
            f.write(f'{item}\n')
        f.write('Val coverage\n')
        for item in val_overlap_results_list:
            f.write(f'{item}\n')

