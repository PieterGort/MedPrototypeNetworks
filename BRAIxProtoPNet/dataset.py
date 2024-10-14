import pandas as pd
import numpy as np
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms.v2 as v2
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None,):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        # convert to RGB slows down the process though
        image = Image.open(img_path).convert('RGB')

        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.df.iloc[idx]['target'])

class MyDatasetAnalysis(Dataset):
    def __init__(self, dataframe, transform=None,):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        # convert to RGB slows down the process though
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.df.iloc[idx]['target']), self.df.iloc[idx]['filepath']

def ISIC_2020_split_train_val(dataframe, val_size=0.25, random_state=42):
    dataframe['filepath'] = dataframe
    gss = GroupShuffleSplit(test_size=val_size, n_splits=1, random_state=random_state)
    train_idx, val_idx = next(gss.split(dataframe, groups=dataframe['patient_id']))
    train_df = dataframe.iloc[train_idx]
    val_df = dataframe.iloc[val_idx]

    return train_df, val_df

def ISIC_split_train_val_test(dataframe, val_size=0.25, test_size=0.2, random_state=42, num_splits=5, k=0):
    # First split the data into train and validation + test sets
    train_val_df, test_df = train_test_split(dataframe, test_size=test_size, stratify=dataframe['target'], random_state=random_state)

    sss = StratifiedKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(sss.split(train_val_df, train_val_df['target'])):
        if i == k:
            break
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    return train_df, val_df, test_df

def VinDrMammo_split_train_val_test(dataframe, random_state=42, num_splits=5, k=0):
    train_val_df = dataframe[dataframe['split'] == 'training']
    test_df = dataframe[dataframe['split'] == 'test']

    skgf = StratifiedGroupKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(skgf.split(train_val_df, y=train_val_df['target'], groups=train_val_df['study_id'])):
        if i == k:
            break

    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    return train_df, val_df, test_df


def get_polyp_df(data_folder, csv_path):
    # Define paths for training and testing CSV files
    train_csv_path = os.path.join(csv_path, "polyp_sentences_train_no_comaBLI.csv")
    test_csv_path = os.path.join(csv_path, "polyp_sentences_test_no_comaBLI.csv")
    
    # Check if the CSV files exist
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"File {train_csv_path} not found")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"File {test_csv_path} not found")
    
    # Read the CSV files
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_df['split'] = 'training'
    test_df['split'] = 'test'

    # merge the two dataframes
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Rename the label column to target
    df = df.rename(columns={'label': 'target'})

    # Add the full file path to the image
    df['path'] = df['path'].apply(lambda x: x.split('/')[-3:])
    df['filepath'] = df['path'].apply(lambda x: os.path.join(data_folder, x[0], x[1], x[2]))

    #drop column path
    df = df.drop(columns=['path'])
    df = df[df['filepath'].apply(os.path.exists)]

    return df

def get_df_vindrmammo(image_dir, csv_path, n_classes):

    df = pd.read_csv(csv_path)
    # breast_df = pd.read_csv(os.path.join(root_dir, 'breast-level_annotations.csv'))

    if n_classes == 2:
        df['breast_birads'] = df['breast_birads'].map({
            'BI-RADS 1': 0,  # negative
            'BI-RADS 2': 0,  # benign
            'BI-RADS 3': 1,  # probably benign FOLLOW UP EXAM NEEDED BI-RADS > 2
            'BI-RADS 4': 1,  # suspicious
            'BI-RADS 5': 1   # highly suggestive of malignancy
        })
    elif n_classes == 3:
        df['breast_birads'] = df['breast_birads'].map({
            'BI-RADS 1': 0,  # negative
            'BI-RADS 2': 1,  # benign
            'BI-RADS 3': 2,  # probably benign
            'BI-RADS 4': 2,  # suspicious
            'BI-RADS 5': 2   # highly suggestive of malignancy
        })
    elif n_classes == 5:
        df['breast_birads'] = df['breast_birads'].map({
            'BI-RADS 1': 0,  # negative
            'BI-RADS 2': 1,  # benign
            'BI-RADS 3': 2,  # probably benign
            'BI-RADS 4': 3,  # suspicious
            'BI-RADS 5': 4   # highly suggestive of malignancy
        })
    else:
        raise NotImplementedError("Specified number of classes not implemented")
    
    # Note to self: breast_birads is used as classification target for the model not the local findings
    df = df.rename(columns={'breast_birads': 'target'})

    # create full file path
    df['filepath'] = df.apply(lambda row: os.path.join(image_dir, row['study_id'], f"{row['image_id']}.png"), axis=1)
    
    # make sure bounding boxes are a list of lists
    df['bounding_boxes'] = df['bounding_boxes'].apply(eval)

    return df

def get_CBIS_DDSM_df(csv_folder):
    train_df = pd.read_csv(os.path.join(csv_folder, 'calc_case_description_train_set.csv'))
    test_df = pd.read_csv(os.path.join(csv_folder, 'calc_case_description_test_set.csv'))

    train_df_files = pd.read_csv(os.path.join(csv_folder, ))

    train_df['filepath'] = train_df['image file path']
    test_df['filepath'] = test_df['image file path']

    train_df['target'] = train_df['pathology'].map({
        'BENIGN_WITHOUT_CALLBACK': 0,
        'BENIGN': 0,
        'MALIGNANT': 1,
    })

    test_df['target'] = test_df['pathology'].map({
        'BENIGN_WITHOUT_CALLBACK': 0,
        'BENIGN': 0,
        'MALIGNANT': 1,
    })

    return train_df, test_df

def split_CBIS_DDSM_train_val(dataframe, random_state=42, num_splits=5, k=0):
    train_val_df = dataframe

    skgf = StratifiedGroupKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(skgf.split(train_val_df, y=train_val_df['target'], groups=train_val_df['patient_id'])):
        if i == k:
            break

    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    return train_df, val_df

def get_df_ISIC(kernel_type, out_dim, data_dir, data_folder, use_meta):

    # 2020 data
    df_train = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'train.csv'))
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/train', f'{x}.jpg'))

    if 'newfold' in kernel_type:
        tfrecord2fold = {
            8:0, 5:0, 11:0,
            7:1, 0:1, 6:1,
            10:2, 12:2, 13:2,
            9:3, 1:3, 3:3,
            14:4, 2:4, 4:4,
        }
    elif 'oldfold' in kernel_type:
        tfrecord2fold = {i: i % 5 for i in range(15)}
    else:
        tfrecord2fold = {
            2:0, 4:0, 5:0,
            1:1, 10:1, 13:1,
            0:2, 9:2, 12:2,
            3:3, 8:3, 11:3,
            6:4, 7:4, 14:4,
        }
    df_train['fold'] = df_train['tfrecord'].map(tfrecord2fold)
    df_train['is_ext'] = 0

    # 2018, 2019 data (external data)
    df_train2 = pd.read_csv(os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}', 'train.csv'))
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}/train', f'{x}.jpg'))
    if 'newfold' in kernel_type:
        df_train2['tfrecord'] = df_train2['tfrecord'] % 15
        df_train2['fold'] = df_train2['tfrecord'].map(tfrecord2fold)
    else:
        df_train2['fold'] = df_train2['tfrecord'] % 5
    df_train2['is_ext'] = 1

    # Preprocess Target
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    if out_dim == 9:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    elif out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
    else:
        raise NotImplementedError()
    
    # concat train data
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    # test data
    df_test = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/test', f'{x}.jpg'))

    # alternative binary class mapping
    df_train['target'] = (df_train['benign_malignant'] == 'malignant').astype(int)
    mel_idx = df_train['target'].value_counts().index.tolist().index(1)
    
    return df_train, df_test, mel_idx

def get_WRsampler(dataframe):
    class_sample_counts = dataframe['target'].value_counts().sort_index().values
    class_weights = 1. / class_sample_counts # about [1, 60] in this case
    sample_weights = torch.FloatTensor([class_weights[t] for t in dataframe['target']])
    WRsampler = WeightedRandomSampler(sample_weights, int(len(sample_weights)), replacement=True)
    return WRsampler

def get_WRsampler_dataset(dataset):
    # Calculate class weights based on the dataset
    targets = [target for _, target in dataset]
    class_sample_counts = np.unique(targets, return_counts=True)[1]
    class_weights = 1. / class_sample_counts
    sample_weights = torch.FloatTensor([class_weights[t] for t in targets])
    WRsampler = WeightedRandomSampler(sample_weights, int(len(sample_weights)), replacement=True)
    return WRsampler

def create_polyp_datasets(root_dir, train_transform, val_transform, push_transform, random_state=42, num_splits=5, k=0):

    full_train_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=val_transform)

    # Get the targets for stratified splitting
    targets = [label for _, label in full_train_dataset.samples]

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    # Split indices for k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        if fold == k:
            break

    # Create training and validation subsets
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    # Apply the validation transform to the validation dataset
    val_dataset.dataset.transform = val_transform

    # Create the train_push dataset using the same dataset but with the push transform
    full_train_dataset.transform = push_transform
    train_push_dataset = Subset(full_train_dataset, train_idx)

    return train_dataset, val_dataset, test_dataset, train_push_dataset

def create_kvasir_datasets(root_dir, train_transform, val_transform, push_transform, random_state=42, num_splits=5, k=0):

    dataset = datasets.ImageFolder(root_dir, transform=None)

    # Get the targets for stratified splitting
    targets = [label for _, label in dataset.samples]

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    # Split indices for k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        if fold == k:
            break

    # Create training and validation subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # apply transforms to the datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create the train_push dataset using the same dataset but with the push transform
    train_push_dataset = Subset(dataset, train_idx)
    train_push_dataset.dataset.transform = push_transform

    return train_dataset, val_dataset, train_push_dataset

def find_corrupted_images(dataset):
    corrupted_images = []
    for index in range(len(dataset)):
        path, _ = dataset.samples[index]
        try:
            # Open the image file
            with Image.open(path) as img:
                # Try to load the image to ensure it is not corrupted
                img.load()
        except (OSError, IOError, Image.DecompressionBombError) as e:
            print(f"Corrupted image found: {path}, error: {e}")
            corrupted_images.append(path)
    return corrupted_images

def get_kvasir_df(kvasir_root):
    images_dir = os.path.join(kvasir_root, 'Kvasir-SEG', 'images')
    masks_dir = os.path.join(kvasir_root, 'Kvasir-SEG', 'masks')
    bbox_dir = os.path.join(kvasir_root, 'Kvasir-SEG', 'bbox')

    train_txt = os.path.join(kvasir_root, 'train.txt')
    val_txt = os.path.join(kvasir_root, 'val.txt')

    # Read the list of images
    with open(train_txt, 'r') as f:
        train_images_lst = [line.strip() for line in f.readlines()]

    with open(val_txt, 'r') as f:
        val_images_lst = [line.strip() for line in f.readlines()]

    # Construct full image paths
    train_images_paths = [os.path.join(images_dir, f'{img}.jpg') for img in train_images_lst]
    val_images_paths = [os.path.join(images_dir, f'{img}.jpg') for img in val_images_lst]

    # Create dataframes for train and val splits
    train_paths_df = pd.DataFrame(train_images_paths, columns=['image_path'])
    val_paths_df = pd.DataFrame(val_images_paths, columns=['image_path'])

    train_paths_df['split'] = 'train'
    val_paths_df['split'] = 'val'

    # Combine train and val dataframes
    df = pd.concat([train_paths_df, val_paths_df], ignore_index=True)

    # Add mask paths and image names
    df['mask_path'] = df['image_path'].apply(lambda x: x.replace('images', 'masks'))
    df['image_name'] = df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # Reorder columns
    df = df[['image_name', 'image_path', 'mask_path', 'split']]
    df = df.rename(columns={'image_path': 'filepath'})

    # Add original image dimensions
    def get_image_dimensions(image_path):
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)

    df[['width', 'height']] = df['filepath'].apply(get_image_dimensions).tolist()

    # Add bounding box data
    bbox_data = []
    for file in os.listdir(bbox_dir):
        if file.endswith('.csv'):
            image_name = os.path.splitext(file)[0]
            bbox_df = pd.read_csv(os.path.join(bbox_dir, file))
            for _, row in bbox_df.iterrows():
                bbox_data.append({
                    'image_name': image_name,
                    'class_name': row['class_name'],
                    'xmin': row['xmin'],
                    'ymin': row['ymin'],
                    'xmax': row['xmax'],
                    'ymax': row['ymax'],
                })

    # rename class_name to target
    for row in bbox_data:
        row['target'] = 1 if row['class_name'] == 'polyp' else 0
    
    # Convert bbox data to a dataframe
    bbox_df = pd.DataFrame(bbox_data)

    # Merge bounding box data with the main dataframe
    df = df.merge(bbox_df, on='image_name', how='left')

    # Normalize the bounding box coordinates with respect to their respective image sizes
    df['xmin'] = df['xmin'] / df['width']
    df['xmax'] = df['xmax'] / df['width']
    df['ymin'] = df['ymin'] / df['height']
    df['ymax'] = df['ymax'] / df['height']

    return df

def save_img_w_bbox(img_path, bboxes, original_dims, save_path='bboxes_example.png', color=(0, 255, 255)):
    """
    Draws bounding boxes on an image and saves the result.

    Args:
        img_path (str): Path to the image file.
        bboxes (list): List of bounding boxes, each specified as [xmin, ymin, xmax, ymax].
        original_dims (tuple): Original dimensions of the image as (height, width).
        save_path (str): Path where the image with bounding boxes will be saved.
        color (tuple): Color of the bounding boxes in BGR format.
    """
    # Read the image using OpenCV
    p_img_bgr = cv2.imread(img_path)
    if p_img_bgr is None:
        print(f"Error: Unable to load image at {img_path}")
        return
    
    original_height, original_width = original_dims
    current_height, current_width = p_img_bgr.shape[:2]

    print(f"Image dimensions (HxW): {current_height, current_width}")

    # Calculate the scaling factors
    x_scale = current_width 
    y_scale = current_height

    # Draw each bounding box on the image
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * x_scale)
        xmax = int(xmax * x_scale)
        ymin = int(ymin * y_scale)
        ymax = int(ymax * y_scale)
        print(f"Drawing bbox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        cv2.rectangle(p_img_bgr, (xmin, ymin), (xmax, ymax), color, thickness=2)
    
    # Convert BGR image to RGB
    p_img_rgb = p_img_bgr[..., ::-1]
    
    # Normalize the image for display
    p_img_rgb = np.float32(p_img_rgb) / 255
    
    # Display the image with bounding boxes
    plt.imshow(p_img_rgb)
    plt.axis('off')
    plt.show()
    
    # Save the image with bounding boxes
    plt.imsave(save_path, p_img_rgb)

def main():

    csv_path = 'finding_annotations_bbox_adjusted.csv'
    image_dir = 'C:/Users/piete/Documents/MScThesisLocal/VinDrMammoTest/processed_images'
    n_classes = 2

    df = get_df_vindrmammo(image_dir, csv_path, n_classes)

    print(df.head())
    print("Unique image ID's:", df['image_id'].nunique())
    print("Unique study ID's:", df['study_id'].nunique())
    print("Unique targets:", df['target'].unique())
    print("Amount of unique targets:", df['target'].nunique())
    print("Unique filepaths:", df['filepath'].nunique())
    bbox_count = df['bounding_boxes'].apply(len).sum()
    print(f"Total number of bounding boxes: {bbox_count}")
    # save to csv
    
    # check for duplicates
    # Do not count the bounding boxes that are filled with [-1, -1, -1, -1]
    bbox_count = df['bounding_boxes'].apply(lambda x: np.array(x) != [-1, -1, -1, -1]).sum()


if __name__ == '__main__':
    main()
