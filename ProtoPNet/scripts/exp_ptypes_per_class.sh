export WANDB_API_KEY=4c3ed5fabc3d875d6d34228648685e12c3182848
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 main.py --experiment_run 'base' \
                                              --gpuid=0 \
                                              --wandb \
                                              --num_workers=18 \
                                              --dataset='ISIC2020' \
                                              --kfold=0 \
                                              --base_architecture='resnet18' \
                                              --img_size=512 \
                                              --val_size=0.25 \
                                              --test_size=0.2 \
                                              --prototype_shape="(20, 128, 1, 1)" \
                                              --num_classes=2 \
                                              --prototype_activation_function='log' \
                                              --add_on_layers_type='regular' \
                                              --train_batch_size=80 \
                                              --test_batch_size=100 \
                                              --train_push_batch_size=75 \
                                              --joint_optimizer_lrs "{'features': 0.0001, 'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --joint_lr_step_size=5 \
                                              --warm_optimizer_lrs "{'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --last_layer_optimizer_lr=0.0001 \
                                              --coefs "{'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 0.0001}" \
                                              --num_train_epochs=50 \
                                              --num_warm_epochs=5 \
                                              --push_start=10

torchrun --nnodes=1 --nproc_per_node=1 main.py --experiment_run '2PClass' \
                                              --gpuid=0 \
                                              --wandb \
                                              --num_workers=18 \
                                              --dataset='ISIC2020' \
                                              --kfold=0 \
                                              --base_architecture='resnet18' \
                                              --img_size=512 \
                                              --val_size=0.25 \
                                              --test_size=0.2 \
                                              --prototype_shape="(4, 128, 1, 1)" \
                                              --num_classes=2 \
                                              --prototype_activation_function='log' \
                                              --add_on_layers_type='regular' \
                                              --train_batch_size=80 \
                                              --test_batch_size=100 \
                                              --train_push_batch_size=75 \
                                              --joint_optimizer_lrs "{'features': 0.0001, 'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --joint_lr_step_size=5 \
                                              --warm_optimizer_lrs "{'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --last_layer_optimizer_lr=0.0001 \
                                              --coefs "{'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 0.0001}" \
                                              --num_train_epochs=50 \
                                              --num_warm_epochs=5 \
                                              --push_start=10

torchrun --nnodes=1 --nproc_per_node=1 main.py --experiment_run '5PClass' \
                                              --gpuid=0 \
                                              --wandb \
                                              --num_workers=18 \
                                              --dataset='ISIC2020' \
                                              --kfold=0 \
                                              --base_architecture='resnet18' \
                                              --img_size=512 \
                                              --val_size=0.25 \
                                              --test_size=0.2 \
                                              --prototype_shape="(10, 128, 1, 1)" \
                                              --num_classes=2 \
                                              --prototype_activation_function='log' \
                                              --add_on_layers_type='regular' \
                                              --train_batch_size=80 \
                                              --test_batch_size=100 \
                                              --train_push_batch_size=75 \
                                              --joint_optimizer_lrs "{'features': 0.0001, 'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --joint_lr_step_size=5 \
                                              --warm_optimizer_lrs "{'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --last_layer_optimizer_lr=0.0001 \
                                              --coefs "{'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 0.0001}" \
                                              --num_train_epochs=50 \
                                              --num_warm_epochs=5 \
                                              --push_start=10

torchrun --nnodes=1 --nproc_per_node=1 main.py --experiment_run '20PClass' \
                                              --gpuid=0 \
                                              --wandb \
                                              --num_workers=18 \
                                              --dataset='ISIC2020' \
                                              --kfold=0 \
                                              --base_architecture='resnet18' \
                                              --img_size=512 \
                                              --val_size=0.25 \
                                              --test_size=0.2 \
                                              --prototype_shape="(40, 128, 1, 1)" \
                                              --num_classes=2 \
                                              --prototype_activation_function='log' \
                                              --add_on_layers_type='regular' \
                                              --train_batch_size=80 \
                                              --test_batch_size=100 \
                                              --train_push_batch_size=75 \
                                              --joint_optimizer_lrs "{'features': 0.0001, 'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --joint_lr_step_size=5 \
                                              --warm_optimizer_lrs "{'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --last_layer_optimizer_lr=0.0001 \
                                              --coefs "{'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 0.0001}" \
                                              --num_train_epochs=50 \
                                              --num_warm_epochs=5 \
                                              --push_start=10

torchrun --nnodes=1 --nproc_per_node=1 main.py --experiment_run '100PClass' \
                                              --gpuid=0 \
                                              --wandb \
                                              --num_workers=18 \
                                              --dataset='ISIC2020' \
                                              --kfold=0 \
                                              --base_architecture='resnet18' \
                                              --img_size=512 \
                                              --val_size=0.25 \
                                              --test_size=0.2 \
                                              --prototype_shape="(200, 128, 1, 1)" \
                                              --num_classes=2 \
                                              --prototype_activation_function='log' \
                                              --add_on_layers_type='regular' \
                                              --train_batch_size=80 \
                                              --test_batch_size=100 \
                                              --train_push_batch_size=75 \
                                              --joint_optimizer_lrs "{'features': 0.0001, 'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --joint_lr_step_size=5 \
                                              --warm_optimizer_lrs "{'add_on_layers': 0.003, 'prototype_vectors': 0.003}" \
                                              --last_layer_optimizer_lr=0.0001 \
                                              --coefs "{'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 0.0001}" \
                                              --num_train_epochs=50 \
                                              --num_warm_epochs=5 \
                                              --push_start=10