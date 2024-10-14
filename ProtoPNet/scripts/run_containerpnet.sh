#!/bin/bash

python main.py --dataset='VinDrMammo' \
                                              --image_folder='C:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammo/Processed_Images_768x768' \
                                              --csv_path='C:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammo/finding_annotations.csv' \
                                              --experiment_run='005' \
                                              --wandb \
                                              --wandb_project='VinDrMammo' \
                                              --gpuid=0 \
                                              --num_workers=8 \
                                              --kfold=0 \
                                              --base_architecture='resnet18' \
                                              --img_height=768 \
                                              --img_width=768 \
                                              --val_size=0.2 \
                                              --test_size=0.2 \
                                              --prototype_shape="(25, 128, 1, 1)" \
                                              --num_classes=5 \
                                              --prototype_activation_function='log' \
                                              --add_on_layers_type='regular' \
                                              --train_batch_size=64 \
                                              --test_batch_size=64 \
                                              --train_push_batch_size=64 \
                                              --joint_optimizer_lrs '{"features": 1e-4, "add_on_layers": 3e-3, "prototype_vectors": 3e-3}' \
                                              --joint_lr_step_size=5 \
                                              --warm_optimizer_lrs '{"add_on_layers": 3e-3, "prototype_vectors": 3e-3}' \
                                              --last_layer_optimizer_lr=1e-4 \
                                              --coefs '{"crs_ent": 1, "clst": 0.8, "sep": -0.08, "l1": 1e-4}' \
                                              --num_train_epochs=50 \
                                              --num_warm_epochs=5 \
                                              --push_start=10 \


