import os
import shutil
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import torchvision.transforms.v2 as v2
import argparse
import re
import wandb
import json

from data.dataset import MyDataset, ISIC_2020_split_train_val, get_WRsampler, get_df_ISIC, \
                                    ISIC_split_train_val_test, get_df_vindrmammo, \
                                    VinDrMammo_split_train_val_test, \
                                    create_polyp_datasets, get_WRsampler_dataset, \
                                    get_CBIS_DDSM_df, split_CBIS_DDSM_train_val, \
                                    create_kvasir_datasets
                                    
from utils.helpers import makedir, create_small_dataloader
import models.model
import models.push
import utils.prune
import train_and_test as tnt
import utils.save
from utils.log import create_logger, log_wandb
from utils.preprocess import preprocess_input_function

def parse_args():
    parser = argparse.ArgumentParser(description="Training a ProtoPNet with custom configurations")

    parser.add_argument('--dataset', type=str, default='VinDrMammo', help='Whether to choose ISIC 2018, 2019 and 2020 or just ISIC 2020')
    parser.add_argument('--image_folder', type=str, default=r'C:\Users\piete\Documents\MScThesisLocal\VinDrMammo\processed_images', help='Path to the image folder')
    parser.add_argument('--csv_path', type=str, default='finding_annotations_bbox_adjusted.csv', help='Path to the csv file')

    parser.add_argument('--experiment_run', type=str, default='test_push', help='Experiment run identifier')
    parser.add_argument('--wandb', action='store_true', default=False, help='Log to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='VinDrMammo', help='Weights & Biases project name')
    parser.add_argument('--gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers for data loading')
    parser.add_argument('--kfold', type=int, default=0)

    parser.add_argument('--base_architecture', type=str, default='resnet18', help='Base architecture')
    parser.add_argument('--img_height', type=int, default=1536, help='Image height')
    parser.add_argument('--img_width', type=int, default=768, help='Image width')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')

    parser.add_argument('--prototype_shape', type=str, default='(14, 128, 1, 1)', help='Prototype shape')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--prototype_activation_function', type=str, default='log', help='Prototype activation function')
    parser.add_argument('--add_on_layers_type', type=str, default='regular', help='Type of add-on layers')
    parser.add_argument('--class_specific', action='store_true', default=True, help='Class specific prototypes')
    parser.add_argument('--activation_percentile', type=float, default=99.43, help='Activation percentile determines the threshold for the bounding box of prototypes')

    # Training parameters
    parser.add_argument('--train_batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=12, help='Testing batch size')
    parser.add_argument('--train_push_batch_size', type=int, default=6, help='Training push batch size')
    parser.add_argument('--joint_optimizer_lrs', type=str, default='{"features": 1e-5, "add_on_layers": 1e-4, "prototype_vectors": 1e-4}', help='Joint optimizer learning rates')
    parser.add_argument('--joint_lr_step_size', type=int, default=5, help='Joint optimizer learning rate step size')
    parser.add_argument('--warm_optimizer_lrs', type=str, default='{"add_on_layers": 1e-4, "prototype_vectors": 1e-4}', help='Warm optimizer learning rates')
    parser.add_argument('--last_layer_optimizer_lr', type=float, default=1e-4, help='Last layer optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--gammaLRscheduler', type=float, default=0.5, help='Gamma for the learning rate scheduler')
    parser.add_argument('--coefs', type=str, default='{"crs_ent": 1, "clst": 0.8, "sep": -0.08, "l1": 1e-4}', help='Loss coefficients')
    parser.add_argument('--num_train_epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--num_warm_epochs', type=int, default=5, help='Number of warm-up epochs')
    parser.add_argument('--push_start', type=int, default=11, help='Epoch to start push operation')
    parser.add_argument('--last_layer_iterations', type=int, default=20, help='Number of last layer iterations')

    parser.add_argument('--save_after_epoch', type=int, default=40, help='Save model after epoch')

    args = parser.parse_args()

    # calculate the epochs to push prototypes
    args.push_epochs = [i for i in range(args.num_train_epochs) if i % 11 == 0]

    # Convert JSON strings to Python objects
    args.prototype_shape = json.loads(args.prototype_shape.replace('(', '[').replace(')', ']'))  # Convert tuple format correctly
    args.joint_optimizer_lrs = json.loads(args.joint_optimizer_lrs)
    args.warm_optimizer_lrs = json.loads(args.warm_optimizer_lrs)
    args.coefs = json.loads(args.coefs)
    args.img_size = (args.img_height, args.img_width)
    
    return args

def main(args):

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    base_architecture_type = re.match('^[a-z]*', args.base_architecture).group(0)
    model_dir = 'exp/saved_models/' + args.dataset + '/' + args.base_architecture + '/' + args.experiment_run + '/'
    print("model_dir is:", model_dir)
    makedir(model_dir)
    
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'models/' + base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'models/model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # Datasets have to contain the following columns: 'image_name', 'target', 'filepath'
    if args.dataset == 'ISIC_combination':
        df, df_test, mel_idx = get_df_ISIC('newfold', 9, args.image_folder, args.img_size, False)
        train_df, val_df, test_df = ISIC_split_train_val_test(df, val_size=0.25, test_size=0.2, random_state=42, num_splits=5, k=args.kfold)
        print("Combined ISIC datasets from 2018, 2019, and 2020")
    
    elif args.dataset == 'ISIC2020':
        args.csv_path = '/gpfs/work5/0/prjs0976/ISIC_2020_Training_JPEG_224_224/ISIC_2020_Training_GroundTruth.csv'
        args.image_folder = '/gpfs/work5/0/prjs0976/ISIC_2020_Training_JPEG_224_224/train'
        df = pd.read_csv(args.csv_path)
        df['filepath'] = df['image_name'].apply(lambda x: os.path.join(args.image_folder, f'{x}.jpg'))
        train_df, val_df = ISIC_2020_split_train_val(df, val_size=args.val_size, random_state=42)
        print("ISIC 2020 Dataset")
    
    elif args.dataset == 'VinDrMammo':
        df = get_df_vindrmammo(args.image_folder, args.csv_path, args.num_classes)
        train_df, val_df, test_df = VinDrMammo_split_train_val_test(df, random_state=42, num_splits=5, k=args.kfold)
        print("VinDr-Mammo Dataset")
        print("Number of classes", args.num_classes)

    elif args.dataset == 'Polyp':
        print("Polyp Dataset")

    elif args.dataset == 'CBIS_DDSM':
        train_df, test_df = get_CBIS_DDSM_df(args.csv_path)
        train_df, val_df = split_CBIS_DDSM_train_val(train_df, random_state=42, num_splits=5, k=args.kfold)
        print("CBIS-DDSM Dataset")

    elif args.dataset == 'Kvasir':
        print("Kvasir Dataset")
    else:
        raise ValueError("Dataset not found. Please choose a valid dataset.")

    # Transformations
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        #v2.RandomResizedCrop(args.img_size, scale=(0.95, 1.0), antialias=True),
        v2.Resize(args.img_size, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        #v2.RandomAffine(degrees=180, translate=(0.10, 0.10), scale=(0.95, 1.05)),
        v2.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.95, 1.05)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(args.img_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    push_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(args.img_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        # No normalization for push operation used in This Looks Like That
    ])

    if args.dataset != 'Polyp' and args.dataset != 'Kvasir':
        train_dataset = MyDataset(train_df, transform=train_transform)
        val_dataset = MyDataset(val_df, transform=val_transform)
        push_dataset = MyDataset(train_df, transform=push_transform)
        test_dataset = MyDataset(test_df, transform=val_transform)

        # create the dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, sampler=get_WRsampler(train_df), shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        train_push_loader = DataLoader(push_dataset, batch_size=args.train_push_batch_size, num_workers=args.num_workers, sampler=get_WRsampler(train_df), shuffle=False, pin_memory=True)
    
    elif args.dataset == 'Polyp':
        # use image folder and csv path to create the dataset
        train_dataset, val_dataset, test_dataset, train_push_dataset = create_polyp_datasets(args.image_folder, train_transform, val_transform, push_transform, k=args.kfold)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, sampler=get_WRsampler_dataset(train_dataset), shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        train_push_loader = DataLoader(train_push_dataset, batch_size=args.train_push_batch_size, num_workers=args.num_workers, sampler=get_WRsampler_dataset(train_push_dataset), shuffle=False, pin_memory=True)
    
    elif args.dataset == 'Kvasir':
        train_dataset, val_dataset, train_push_dataset = create_kvasir_datasets(args.image_folder, train_transform, val_transform, push_transform, random_state=42, num_splits=5, k=args.kfold)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, sampler=get_WRsampler_dataset(train_dataset), shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        train_push_loader = DataLoader(train_push_dataset, batch_size=args.train_push_batch_size, num_workers=args.num_workers, sampler=get_WRsampler_dataset(train_push_dataset), shuffle=False, pin_memory=True)      

    else:
        raise ValueError("Dataset not found. Please choose a valid dataset.")
    
    # construct the model
    ppnet = models.model.construct_PPNet(base_architecture=args.base_architecture,
                                pretrained=True, 
                                img_size=args.img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.num_classes,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type)
    
    if args.prototype_activation_function == 'linear':
       ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    # make sure the model is on the GPU either a single or multiple
    ppnet = ppnet.cuda(device=device)
    # ppnet_multi = DistributedDataParallel(ppnet, device_ids=[args.gpuid])
    ppnet_multi = torch.nn.DataParallel(ppnet)
    ppnet_multi = ppnet_multi.cuda()
    class_specific = args.class_specific

    # # Count the number of model parameters to be trained
    # count_param = 0
    # for _, param in ppnet.named_parameters():
    #     if param.requires_grad:
    #         count_param+=1
    # pytorch_total_params = sum(p.numel() for p in ppnet.parameters() if p.requires_grad)
    # log('Total number of trainable parameters: {}'.format(pytorch_total_params))
    # log('Number of layers that require gradient: \t{0}'.format(count_param))

    # define optimizer
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': args.joint_optimizer_lrs['features'], 'weight_decay': args.weight_decay}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': args.joint_optimizer_lrs['add_on_layers'], 'weight_decay': args.weight_decay},
    {'params': ppnet.prototype_vectors, 'lr': args.joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=args.joint_lr_step_size, gamma=args.gammaLRscheduler)

    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': args.warm_optimizer_lrs['add_on_layers'], 'weight_decay': args.weight_decay},
    {'params': ppnet.prototype_vectors, 'lr': args.warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': args.last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    log('start training')
    global_step = 0
    for epoch in range(args.num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < args.num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            train_scores = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=args.coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            train_scores = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=args.coefs, log=log)
            
            # LEARNING RATE UPDATE
            joint_lr_scheduler.step()
        
        log_wandb('train', train_scores, global_step, args.num_classes)

        val_scores = tnt.val(model=ppnet_multi, dataloader=val_loader,
                        class_specific=class_specific, log=log, coefs=args.coefs)
        log_wandb('val', val_scores, global_step, args.num_classes)       
        global_step += 1

        utils.save.save_best_model(model=ppnet,
                                   model_dir=model_dir,
                                   model_name=str(epoch) + 'nopush',
                                   scores=val_scores,
                                   current_epoch=epoch,
                                   after_epoch=args.save_after_epoch, log=log)

        if epoch >= args.push_start and epoch in args.push_epochs:
            models.push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log,
                activation_percentile=args.activation_percentile,)
            
            push_val_scores = tnt.val(model=ppnet_multi, dataloader=val_loader,
                            class_specific=class_specific, log=log, coefs=args.coefs)
            log_wandb('val', push_val_scores, global_step, args.num_classes)
            global_step += 1

            utils.save.save_best_model(model=ppnet,
                                   model_dir=model_dir,
                                   model_name=str(epoch) + 'push',
                                   scores=push_val_scores,
                                   current_epoch=epoch,
                                   after_epoch=args.save_after_epoch, log=log)

            if args.prototype_activation_function != 'linear':

                tnt.last_only(model=ppnet_multi, log=log)

                for i in range(args.last_layer_iterations):
                    log('iteration: \t{0}'.format(i))
                    ll_train_scores = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=args.coefs, log=log)
                    log_wandb('train', ll_train_scores, global_step, args.num_classes)        
                    ll_val_scores = tnt.val(model=ppnet_multi, dataloader=val_loader,
                                    class_specific=class_specific, log=log, coefs=args.coefs)

                    log_wandb('val', ll_val_scores, global_step, args.num_classes)
                    global_step += 1

                    utils.save.save_best_model(model=ppnet,
                                                        model_dir=model_dir,
                                                        model_name=str(epoch) + 'push',
                                                        scores=ll_val_scores,
                                                        current_epoch=epoch,
                                                        after_epoch=args.save_after_epoch, log=log)
                    
    torch.save(ppnet, os.path.join(model_dir, 'final_model.pth'))
    logclose()

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

if __name__ == '__main__':
    args = parse_args()
    config_dict = vars(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpuid) # '0,1,2,3
    print("Using CUDA_VISIBLE_DEVICES =", os.environ['CUDA_VISIBLE_DEVICES'])

    if args.wandb:
        wandb.init(project=args.wandb_project, config=config_dict)
        wandb.run.name = args.base_architecture + '_' + args.experiment_run
        wandb.save('main.py')
        wandb.save('train_and_test.py')
        wandb.save('data/dataset.py')
    else:
        wandb.init(project=args.wandb_project, config=config_dict, mode='disabled')
    main(args)
    if args.wandb:
        wandb.finish()

