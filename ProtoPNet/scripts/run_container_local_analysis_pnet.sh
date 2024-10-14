

modeldir="/home/pgort/vscodethesis/ProtoPNet/saved_models/resnet18/val_loader"
model="20push0.7836.pth"
imgdir="/gpfs/work5/0/prjs0976/ISIC_2020_Training_JPEG/train"
img="ISIC_0149568.jpg"
imgclass=1


torchrun --nnodes=1 --nproc_per_node=1 local_analysis.py \
                                        --modeldir=$modeldir \
                                        --model=$model \
                                        --imgdir=$imgdir \
                                        --img=$img \
                                        --imgclass=$imgclass \
                                       
