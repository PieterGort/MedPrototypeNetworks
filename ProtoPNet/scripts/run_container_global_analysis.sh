
gpuid=0
image_folder="/gpfs/work5/0/prjs0976/ISIC_2020_Training_JPEG/train"
labels_file="/gpfs/work5/0/prjs0976/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth.csv"
modeldir="/home/pgort/ProtoPNet/exp/saved_models/resnet18/10_per_class"
model="20push0.7558.pth"


torchrun --nnodes=1 --nproc_per_node=1 global_analysis.py \
                                        --gpuid=$gpuid \
                                        --image_folder=$image_folder \
                                        --labels_file=$labels_file \
                                        --modeldir=$modeldir \
                                        --model=$model \
                                       
