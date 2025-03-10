##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
import re
import pandas as pd

import os
import copy

from data.dataset import get_df_ISIC, ISIC_split_train_val_test
from utils.helpers import makedir, find_high_activation_crop
from utils.log import create_logger
from utils.preprocess import mean, std, undo_preprocess_input_function

import argparse

def main():

    local_analysis_path = 'ProtoPNet/exp/local_analysis/'
    model_base_architecture = modeldir.split('/')[-2]
    experiment_run = '/'.join(modeldir.split('/')[-1:])
    save_analysis_path = os.path.join(local_analysis_path, model_base_architecture, experiment_run, model)

    if not os.path.exists(save_analysis_path):
        os.makedirs(save_analysis_path)

    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    log('Save path: ' + save_analysis_path)

    # # load the dataframe and print the actual label
    # df, _, _ = get_df('newfold', 9, image_folder, 512, False)
    # train_df, _, test_df = split_train_val_test(df, val_size=0.25, test_size=0.2)

    df, _, _ = get_df_ISIC('newfold', 9, args.image_folder, 512, False)
    train_df, val_df, test_df = ISIC_split_train_val_test(df, val_size=0.25, test_size=0.2, random_state=42, num_splits=5, k=0)

    print(args.img_path)
    print(df['filepath'][0])

    image_row = df[df['filepath'] == args.img_path]

    
    # image_row = train_df[test_df['filepath'] == test_image_dir]
    test_image_label = image_row['target'].astype(int).values[0]

    if test_image_label == 0:
        log("The image has label 0, which is benign.")
    else:
        log("The image has label 1, which is malignant.")

    epoch_number_str = re.search(r'\d+', args.model).group(0)
    start_epoch_number = int(epoch_number_str)

    ppnet = torch.load(os.path.join(args.modeldir, args.model))
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    normalize = transforms.Normalize(mean=mean, std=std)

    ##### SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(args.modeldir, 'img')
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
    prototype_img_identity = prototype_info[:, -1]

    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    log('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')

    ##### HELPER FUNCTIONS FOR PLOTTING
    def save_preprocessed_img(fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
        undo_preprocessed_img = undo_preprocess_input_function(img_copy)
        print('image index {0} in batch'.format(index))
        undo_preprocessed_img = undo_preprocessed_img[0]
        undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
        undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
        
        plt.imsave(fname, undo_preprocessed_img)
        return undo_preprocessed_img

    def save_prototype(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)
        
    def save_prototype_self_activation(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                        'prototype-img-original_with_self_act'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

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

    # load the test image and forward it through the network
    preprocess = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    normalize
    ])

    img_pil = Image.open(args.img_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])

    logits, min_distances = ppnet_multi(images_test)
    conv_output, distances = ppnet.push_forward(images_test)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    tables = []
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
        log(str(i) + ' ' + str(tables[-1]))

    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    log('Predicted: ' + str(predicted_cls))
    log('Actual: ' + str(correct_cls))
    original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                        images_test, idx)

    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

    log('Most activated 10 prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    for i in range(1,11):
        log('top {0} activated prototype for this image:'.format(i))
        save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'top-%d_activated_prototype.png' % i),
                    start_epoch_number, sorted_indices_act[-i].item())
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                                'top-%d_activated_prototype_in_original_pimg.png' % i),
                                            epoch=start_epoch_number,
                                            index=sorted_indices_act[-i].item(),
                                            bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                            bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                            bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                            bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                            color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                    'top-%d_activated_prototype_self_act.png' % i),
                                    start_epoch_number, sorted_indices_act[-i].item())
        log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
        log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
        if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
        log('activation value (similarity score): {0}'.format(array_act[-i]))
        log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
        
        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile=args.activation_map_threshold)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                overlayed_img)
        log('--------------------------------------------------------------')

    ##### PROTOTYPES FROM TOP-k CLASSES
    k = 2
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

        log('top %d predicted class: %d' % (i+1, c))
        log('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'top-%d_activated_prototype.png' % prototype_cnt),
                        start_epoch_number, prototype_index)
            save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                    'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                epoch=start_epoch_number,
                                                index=prototype_index,
                                                bbox_height_start=prototype_info[prototype_index][1],
                                                bbox_height_end=prototype_info[prototype_index][2],
                                                bbox_width_start=prototype_info[prototype_index][3],
                                                bbox_width_end=prototype_info[prototype_index][4],
                                                color=(0, 255, 255))
            save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                        'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                        start_epoch_number, prototype_index)
            log('prototype index: {0}'.format(prototype_index))
            log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
            log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
            
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                    interpolation=cv2.INTER_CUBIC)
            
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                    high_act_patch)
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                            img_rgb=original_img,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                    overlayed_img)
            log('--------------------------------------------------------------')
            prototype_cnt += 1
        log('***************************************************************')

    if predicted_cls == correct_cls:
        log('Prediction is correct.')
    else:
        log('Prediction is wrong.')

    logclose()

if __name__ == '__main__':
    sys.path.insert(0, './models')
    image_path = 'C:/Users/piete/Documents/MScThesisLocal/ISIC\jpeg-melanoma-512x512/train\ISIC_0156460.jpg'
    modeldir="exp/saved_models/resnet18/base_512_#2"
    model = "40push0.8979.pth"
    csv_path = 'C:/Users/piete/Documents/MScThesisLocal/ISIC/jpeg-melanoma-512x512/train.csv'

    image_folder = 'C:/Users/piete/Documents/MScThesisLocal/ISIC'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--modeldir', type=str, default=modeldir)
    parser.add_argument('--model', type=str, default=model)
    parser.add_argument('--img_path', type=str, default=image_path)
    parser.add_argument('--image_folder', type=str, default=image_folder)
    parser.add_argument('--csv_path', type=str, default=csv_path)
    parser.add_argument('--activation_map_threshold', type=float, default=97)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    main()