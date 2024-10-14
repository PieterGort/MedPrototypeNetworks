import numpy as np

resfield = r'saved_models\Polyp\resnet18\5fCV\img_24_8\epoch-60\bb-receptive_field60.npy'
bb = r'saved_models\Polyp\resnet18\5fCV\img_24_8\epoch-60\bb60.npy'

bb = np.load(bb)

print(bb[16, :])