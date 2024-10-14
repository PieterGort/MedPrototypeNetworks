import os
import shutil

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transformsv2
import torchvision.datasets as datasets

import argparse
import sys

from utils.helpers import makedir
import models.push
import models.model
import utils.prune as prune
import train_and_test as tnt
import utils.save as save
from utils.log import create_logger
from utils.preprocess import mean, std, preprocess_input_function
from data.dataset import MyDataset, MyDatasetAnalysis, ISIC_2020_split_train_val, get_WRsampler, get_df_ISIC, \
                                    ISIC_split_train_val_test, get_df_vindrmammo, VinDrMammo_split_train_val_test, create_polyp_datasets, get_WRsampler_dataset

def main():
    optimize_last_layer = args.optimize_last_layer
    # pruning parameters
    k = args.topk
    prune_threshold = args.prune_threshold

    original_model_dir = args.modeldir #"exp/saved_models/resnet18/base_512"
    original_model_name = args.model #"38nopush0.8612.pth"

    need_push = ('nopush' in original_model_name)
    if need_push:
        assert(False) # pruning must happen after push
    else:
        epoch = original_model_name.split('push')[0]

    if '_' in epoch:
        epoch = int(epoch.split('_')[0])
    else:
        epoch = int(epoch)

    model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch,
                                            k,
                                            prune_threshold))
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

    # DATASET
    df = get_df_vindrmammo(args.image_folder, args.csv_path, args.num_classes)
    train_df, _, test_df = VinDrMammo_split_train_val_test(df, random_state=42, num_splits=5, k=args.kfold)
    
    WRsampler = get_WRsampler(train_df)

    train_transform = transformsv2.Compose([
    transformsv2.Resize(args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    push_transform = transformsv2.Compose([
    transformsv2.Resize(args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    ])

    test_transform = transformsv2.Compose([
    transformsv2.Resize(args.img_size),
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MyDataset(train_df, transform=train_transform)
    train_push_dataet = MyDataset(train_df, transform=push_transform)
    test_dataset = MyDataset(test_df, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=WRsampler, num_workers=args.num_workers)
    train_push_loader = DataLoader(train_push_dataet, batch_size=args.batch_size, sampler=WRsampler, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    ppnet = torch.load(original_model_dir + '/' + original_model_name)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    coefs = {
                'crs_ent': 1,
                'clst': 0.8,
                'sep': -0.08,
                'l1': 1e-4,
            }
    
    tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs,
            class_specific=class_specific, log=log)

    # prune prototypes
    log('prune')
    prune.prune_prototypes(dataloader=train_push_loader,
                        prototype_network_parallel=ppnet_multi,
                        k=k,
                        prune_threshold=prune_threshold,
                        preprocess_input_function=preprocess_input_function, # normalize
                        original_model_dir=original_model_dir,
                        epoch_number=epoch,
                        #model_name=None,
                        log=log,
                        copy_prototype_imgs=True)
    
    scores_prune = tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs,
                    class_specific=class_specific, log=log)
    
    auroc_prune = scores_prune['AUROC']

    save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                model_name=original_model_name.split('push')[0] + 'prune',
                                metric=auroc_prune,
                                target_value=0.83, log=log)

    # last layer optimization
    if optimize_last_layer:
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        log('optimize last layer')
        tnt.last_only(model=ppnet_multi, log=log)
        for i in range(50):
            log('iteration: \t{0}'.format(i))
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
            scores_ll = tnt.val(model=ppnet_multi, dataloader=test_loader, coefs=coefs,
                            class_specific=class_specific, log=log)
            save.save_best_model(model=ppnet,
                                model_dir=model_dir,
                                model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                scores=scores_ll,
                                current_epoch=i,
                                after_epoch=0, log=log)

    logclose()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--image_folder', type=str, default=r'C:\Users\piete\Documents\MScThesisLocal\VinDrMammo\Processed_Images')
    parser.add_argument('--csv_path', type=str, default=r'C:\Users\piete\Documents\MScThesisLocal\VinDrMammo\finding_annotations.csv')
    parser.add_argument('--modeldir', type=str, default='exp/saved_models/VinDrMammo/resnet34/Classes_00111')
    parser.add_argument('--model', type=str, default="44push0.8024.pth")
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_height', type=int, default=1536)
    parser.add_argument('--img_width', type=int, default=768)
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--optimize_last_layer', type=bool, default=True)
    parser.add_argument('--prune_threshold', type=float, default=3)

    args = parser.parse_args()
    args.img_size = args.img_height, args.img_width
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    main()