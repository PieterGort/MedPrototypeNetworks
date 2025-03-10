import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
from operator import itemgetter

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    device=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes

    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    
    #saves the patch information of the closest distance patch 
    global_min_proto_info = np.zeros([n_prototypes, 5])

    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    min_each_image_proto_info = {}

    num_batches = len(dataloader)

    print("storing prototype distance to all images..", flush=True)
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        print("processing batch no:{}/{}".format(push_iter, num_batches), flush=True)

        start_index_of_search_batch = push_iter * search_batch_size

        store_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   min_each_image_proto_info, # this will be updated
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   device=device)
        
    print("saving the closest image per prototype globally..", flush=True)
    prototype_selection_greedy(prototype_network_parallel, min_each_image_proto_info, n_prototypes, global_min_proto_dist, global_min_proto_info)
    
    print("storing global prototype patches from the images", flush=True)
    visualize_update_proto(global_min_proto_info, 
                    dataloader, 
                    search_batch_size, 
                    prototype_network_parallel, 
                    preprocess_input_function, 
                    prototype_layer_stride, 
                    proto_rf_boxes, # this will be updated
                    proto_bound_boxes, # this will be updated 
                    global_min_fmap_patches, # this will be updated
                    num_batches,
                    epoch_number,
                    dir_for_saving_prototypes=proto_epoch_dir,
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                    device=device)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    print("Number of zero in the global min fmap patches. There should not be any if the code is correct:", np.count_nonzero(global_min_fmap_patches==0))
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(device)) #cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def store_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               min_each_image_proto_info,
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               device=None):
    '''
    min_each_image_proto_info: 
    dictionary; 
    key - prototype id, 
    value - global img idx, min dist of the prototype to an img, min y, min x of the patch in the feature map. 
    '''
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.to(device) #cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    #print("proto dist:", proto_dist_.shape) #12, 400, 95, 47
    for j in range(n_prototypes):
        #print("prototype number:", j)
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
            #print("proto dist:", proto_dist_.shape)
            #print("proto dist j:", proto_dist_j)
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:]
        
        #print("proto dist j:", proto_dist_j.shape) #8, 95, 47
        batch_min_proto_dist_j = np.amin(proto_dist_j.reshape(proto_dist_j.shape[0], -1), axis=1) # min in each image
        #print(all_min_proto_dist_j_dic[str(j)])
        #print("all min shape:", all_min_proto_dist_j_dic[str(j)].shape)
        min_idx = np.argmin(proto_dist_j.reshape(proto_dist_j.shape[0], -1), axis=1)
        unravel_min_idx = list(np.unravel_index(min_idx, proto_dist_j[0,:,:].shape))
        #print("unravel min:", unravel_min_idx)
        #print("start index:", start_index_of_search_batch)
        #print("batch index:", np.array(class_to_img_index_dict[target_class]))
        #print("training set index:", np.array(class_to_img_index_dict[target_class]) + start_index_of_search_batch)
        batch_img_idx = np.array(class_to_img_index_dict[target_class]) + start_index_of_search_batch
        batch_min_each_image_info_proto_j = np.array(list(zip(batch_img_idx, batch_min_proto_dist_j, unravel_min_idx[0], unravel_min_idx[1])))
        #print("batch min shape:", batch_min_each_image_info_proto_j.shape)
        if str(j) not in min_each_image_proto_info.keys():
            min_each_image_proto_info[str(j)] = batch_min_each_image_info_proto_j
        else:
            min_each_image_proto_info[str(j)] = np.append(min_each_image_proto_info[str(j)], batch_min_each_image_info_proto_j, axis = 0)
        #print("appending batch:", min_each_image_proto_info[str(j)].shape)

def prototype_selection_greedy(prototype_network_parallel, min_each_image_proto_info, n_prototypes, global_min_proto_dist, global_min_proto_info):
    num_classes = prototype_network_parallel.module.num_classes
    if num_classes == 2:
        images_assigned_idx = {0:[], 1:[]}
    elif num_classes == 5:
        images_assigned_idx = {0:[], 1:[], 2:[], 3:[], 4:[]}
    #print("min each image proto info:", min_each_image_proto_info)
    #print("min each image proto info proto 0 shape:", min_each_image_proto_info['0'].shape)
    for j in range(n_prototypes):
        target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
        #print("prototype:", j)
        #print("prototype details shape before deletion:", min_each_image_proto_info[str(j)].shape)
        #print("sorted array:", sorted(min_each_image_proto_info[str(j)], key=itemgetter(1))[:10])
        if images_assigned_idx[target_class]:
            min_each_image_proto_info[str(j)] = min_each_image_proto_info[str(j)][~np.isin(min_each_image_proto_info[str(j)][:,0], images_assigned_idx[target_class])]
            #print("prototype details shape after deletion:", min_each_image_proto_info[str(j)].shape)
            #print("have the index been deleted correctly?", np.isin(min_each_image_proto_info[str(j)][:,0], images_assigned_idx[target_class]).any())
        min_idx = np.argmin(min_each_image_proto_info[str(j)][:, 1])
        min_val = min_each_image_proto_info[str(j)][min_idx]
        #print("min idx, val:", min_idx, min_val)
        global_min_proto_info[j] = np.concatenate((np.array([j]), min_val))
        global_min_proto_dist[j] = min_val[1]
        images_assigned_idx[target_class].append(min_val[0])
        #print("images assigned:", images_assigned_idx[target_class])
        #print("shape after appending idx:", global_min_proto_info[j].shape)
        #input('halt')
    
    del images_assigned_idx

def visualize_update_proto(global_min_proto_info, 
                    dataloader, 
                    search_batch_size, 
                    prototype_network_parallel, 
                    preprocess_input_function, 
                    prototype_layer_stride, 
                    proto_rf_boxes, # this will be updated
                    proto_bound_boxes, # this will be updated 
                    global_min_fmap_patches, # this will be updated 
                    num_batches,
                    epoch_number,
                    dir_for_saving_prototypes=None,
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    prototype_activation_function_in_numpy=None,
                    device=None):
    
    prototype_network_parallel.eval()
    proto_ids = []
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        
        print("visualize batch no:{}/{}".format(push_iter, num_batches), flush=True)

        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input
        
        start_index_of_search_batch = push_iter * search_batch_size
        #print('start index of search batch:', start_index_of_search_batch)
        matched_proto = global_min_proto_info[(global_min_proto_info[:,1] >= start_index_of_search_batch) & (global_min_proto_info[:,1] < (start_index_of_search_batch+search_batch_size))]
        
        #print("matched proto:", matched_proto)

        if len(matched_proto):
            #print("actual image index in the training set:", matched_proto[:,1])
            matched_proto[:,1] = matched_proto[:,1] - start_index_of_search_batch
            with torch.no_grad():
                search_batch = search_batch.to(device)
                protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)#_input[matched_proto[:,0]])
        
            protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
            proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

            del protoL_input_torch, proto_dist_torch

            prototype_shape = prototype_network_parallel.module.prototype_shape
            n_prototypes = prototype_shape[0]
            proto_h = prototype_shape[2]
            proto_w = prototype_shape[3]
            max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        for k in range(matched_proto.shape[0]):
            # retrieve the corresponding feature map patch
            j = int(matched_proto[k, 0]) #prototype id
            proto_ids.append(j)
            img_index_in_batch = int(matched_proto[k, 1])
            #print("prototype:", j)
            #print("img index in batch:", img_index_in_batch)
            fmap_height_start_index = int(matched_proto[k, 3]) * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = int(matched_proto[k, 4]) * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
            #print(fmap_height_start_index, fmap_height_end_index)
            #print(fmap_width_start_index, fmap_width_end_index)

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                :,
                                                fmap_height_start_index:fmap_height_end_index,
                                                fmap_width_start_index:fmap_width_end_index]
            
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            matched_proto_info = np.array([int(matched_proto[k, 1]), int(matched_proto[k, 3]), int(matched_proto[k, 4])])
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), matched_proto_info, protoL_rf_info)

            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = [original_img_j.shape[0], original_img_j.shape[1]]
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                    rf_prototype_j[3]:rf_prototype_j[4], :]
            #print("proto rf boxes:", rf_prototype_j)
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'exp':
                temp = 128.
                proto_act_img_j = np.exp(-proto_dist_img_j/temp)
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            
            #print("proto_dist_img_j:", proto_dist_img_j, flush=True)
            #print("prot_act_img_j:", proto_act_img_j, flush=True)
            
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size[1], original_img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
            #print("upsample_act_img_j:", upsampled_act_img_j, flush=True)
            
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j, percentile = 99.9)
            #print("proto bound j:", proto_bound_j, flush=True)
            
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                        proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if epoch_number%60==0:
                if dir_for_saving_prototypes is not None:
                    if prototype_self_act_filename_prefix is not None:
                        # save the numpy array of the prototype self activation
                        np.save(os.path.join(dir_for_saving_prototypes,
                                            prototype_self_act_filename_prefix + str(j) + '.npy'),
                                proto_act_img_j)
                    if prototype_img_filename_prefix is not None:
                        # save the whole image containing the prototype as png
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                                original_img_j,
                                vmin=0.0,
                                vmax=1.0)
                        # overlay (upsampled) self activation on original image and save the result
                        rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[...,::-1]
                        overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                                overlayed_original_img_j,
                                vmin=0.0,
                                vmax=1.0)
                        
                        # if different from the original (whole) image, save the prototype receptive field as png
                        if rf_img_j.shape[0] != original_img_size[0] or rf_img_j.shape[1] != original_img_size[1]:
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                    rf_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                            overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                        rf_prototype_j[3]:rf_prototype_j[4]]
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                    overlayed_rf_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                        
                        # save the prototype image (highly activated region of the whole image)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + str(j) + '.png'),
                                proto_img_j,
                                vmin=0.0,
                                vmax=1.0)
    
    print("number of updated prototypes:", np.unique(proto_ids).shape[0], flush=True)
    print("numver of total prototypes:", n_prototypes, flush=True)

    assert np.unique(proto_ids).shape[0] == n_prototypes
    
    del matched_proto
    del matched_proto_info
    del proto_ids

        