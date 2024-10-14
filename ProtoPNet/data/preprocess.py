import os
import pandas as pd
import pydicom
import cv2
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
from concurrent.futures import ProcessPoolExecutor
import time
import json
import zipfile

def read_dicom_from_zip(zip_path):
    dicom_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.dicom'):
                dicom_files.append(file)
    return dicom_files

# Step 2: Scanning for DICOM files in the directory
def scantree(path, extention=None):
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path, extention)
        else:
            if extention != None and os.path.splitext(entry.path)[1] == extention:
                yield entry.path
            elif extention == None:
                yield entry.path

# Step 3: Fit the image by cropping and adjust bounding boxes
def fit_image(fname, bounding_boxes_df, save_processed_path):
    try:
        dicom = pydicom.dcmread(fname)
        X = apply_voi_lut(dicom.pixel_array, dicom, prefer_lut=False)
    except Exception as e:
        print(f"File {fname} raised an exception: {str(e)}")
        return None
    
    # Normalize the image
    X = (X - X.min()) / (X.max() - X.min())
    
    # Handle MONOCHROME1 images
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        X = 1 - X
    
    X = X * 255

    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    X = X[10:-10, 10:-10]

    # regions of non-empty pixels
    output = cv2.connectedComponentsWithStats((X > 20).astype(np.uint8), 8, cv2.CV_32S)

    # stats.shape == (N, 5), where N is the number of regions, 5 dimensions correspond to:
    # left, top, width, height, area_size
    stats = output[2]

    # finding max area which always corresponds to the breast data. 
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h

    # Crop the image to the breast tissue region
    X_fit = X[y1:y2, x1:x2]

    # Adjust bounding boxes to the cropped region
    study_id = fname.split('\\')[-2]
    image_id = fname.split('\\')[-1].split('.')[0]
    boxes = bounding_boxes_df[(bounding_boxes_df['study_id'] == study_id) & (bounding_boxes_df['image_id'] == image_id)]
    adjusted_bboxes = []

    # Adjust bounding boxes to the cropped region
    for _, box in boxes.iterrows():
        xmin = max(box['xmin'] - x1 -10, 0)
        ymin = max(box['ymin'] - y1 -10, 0)
        xmax = max(box['xmax'] - x1 -10, 0)
        ymax = max(box['ymax'] - y1 -10, 0)
        adjusted_bboxes.append([xmin, ymin, xmax, ymax])

    # Save the processed image
    # save_dir = f'C:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammoOriginal/Processed_Images/{study_id}'
    save_dir = os.path.join(save_processed_path, study_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{image_id}.png')
    cv2.imwrite(save_path, X_fit)

    return study_id, image_id, adjusted_bboxes, h, w

def fit_image_zip_path(zip_path, fname, bounding_boxes_df, save_processed_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(fname) as file:
                dicom = pydicom.dcmread(file)
                X = apply_voi_lut(dicom.pixel_array, dicom, prefer_lut=False)

        # dicom = pydicom.dcmread(fname)
        # X = apply_voi_lut(dicom.pixel_array, dicom, prefer_lut=False)
    except Exception as e:
        print(f"File {fname} raised an exception: {str(e)}")
        return None
    
    # Normalize the image
    X = (X - X.min()) / (X.max() - X.min())

    # Handle MONOCHROME1 images
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        X = 1 - X

    X = X * 255

    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    X = X[10:-10, 10:-10]

    # regions of non-empty pixels
    output = cv2.connectedComponentsWithStats((X > 20).astype(np.uint8), 8, cv2.CV_32S)

    # stats.shape == (N, 5), where N is the number of regions, 5 dimensions correspond to:
    # left, top, width, height, area_size
    stats = output[2]

    # finding max area which always corresponds to the breast data. 
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h

    # Crop the image to the breast tissue region
    X_fit = X[y1:y2, x1:x2]

    # Adjust bounding boxes to the cropped region
    study_id = fname.split('/')[-2]
    image_id = fname.split('/')[-1].split('.')[0]
    boxes = bounding_boxes_df[(bounding_boxes_df['study_id'] == study_id) & (bounding_boxes_df['image_id'] == image_id)]
    adjusted_bboxes = []

    # Adjust bounding boxes to the cropped region
    for _, box in boxes.iterrows():
        xmin = max(box['xmin'] - x1 -10, 0)
        ymin = max(box['ymin'] - y1 -10, 0)
        xmax = max(box['xmax'] - x1 -10, 0)
        ymax = max(box['ymax'] - y1 -10, 0)
        adjusted_bboxes.append([xmin, ymin, xmax, ymax])

    # Save the processed image
    # save_dir = f'C:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammoOriginal/Processed_Images/{study_id}'
    save_dir = os.path.join(save_processed_path, study_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{image_id}.png')
    cv2.imwrite(save_path, X_fit)

    return study_id, image_id, adjusted_bboxes, h, w


# Step 4: Batch process all images
def fit_all_images(zip_path, all_images, bounding_boxes_df):
    with ProcessPoolExecutor() as p:
        results = list(p.map(fit_image_parallel, [(zip_path, file, bounding_boxes_df) for file in all_images]))

    return results

# Helper function to call fit_image for parallel processing
def fit_image_parallel(args):
    return fit_image(*args)

# Step 5: Save adjusted bounding boxes to a CSV
def save_adjusted_bboxes(results):
    bbox_list = []
    for study_id, image_id, bboxes, height, width in results:
        for bbox in bboxes:
            bbox_list.append([study_id, image_id] + bbox + [height, width])
    
    bbox_df = pd.DataFrame(bbox_list, columns=['study_id', 'image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width'])
    # save the bounding boxes to a csv file
    bbox_df.to_csv('bbox_df.csv', index=False)

    return bbox_df

def adjust_bounding_boxes(find_df, bbox_df):

    # Merge the finding annotations with the bounding boxes
    find_df_bbox_adjusted = find_df.merge(bbox_df, on=['study_id', 'image_id'], suffixes=('', '_adjusted'))

    # update bounding box columns in find_df_bbox_adjusted
    for col in ['xmin', 'ymin', 'xmax', 'ymax', 'height', 'width']:
        find_df_bbox_adjusted[col] = find_df_bbox_adjusted[col + '_adjusted']
        find_df_bbox_adjusted.drop(col + '_adjusted', axis=1, inplace=True)

    find_df_bbox_adjusted.drop_duplicates(inplace=True)
    df = find_df_bbox_adjusted

    df['width'] = df['width'].replace(0, np.nan)
    df['height'] = df['height'].replace(0, np.nan)

    # Normalize the bounding boxes with respect to the image size
    df['xmin'] = df['xmin'] / df['width']
    df['ymin'] = df['ymin'] / df['height']
    df['xmax'] = df['xmax'] / df['width']
    df['ymax'] = df['ymax'] / df['height']

    df[['xmin', 'xmax', 'ymin', 'ymax']] = df[['xmin', 'xmax', 'ymin', 'ymax']].fillna(-1)

    # Remove duplicate bounding boxes
    df.drop_duplicates(subset=['study_id', 'image_id', 'xmin', 'ymin', 'xmax', 'ymax'], inplace=True)

    # create bounding boxes
    df['bounding_boxes'] = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

    # group the dataframe by study_id and image_id and make sure the bounding boxes are stored as a list of lists
    grouped_df = df.groupby(['study_id', 'image_id']).agg({
        'bounding_boxes': lambda x: x.tolist(),
        'breast_birads': 'first',
        'height': 'first',
        'width': 'first',
        'split': 'first',
    }).reset_index()

    # Convert list of lists to JSON format for saving to CSV
    grouped_df['bounding_boxes'] = grouped_df['bounding_boxes'].apply(json.dumps)

    grouped_df.drop_duplicates(inplace=True)

    return grouped_df

# def main():
#     start = time.time()
#     csv_file_path = 'C:/Users/piete/Documents/MScThesisLocal/VinDrMammoTest/finding_annotations.csv'
#     save_csv_path = 'C:/Users/piete/Documents/MScThesisLocal/VinDrMammoTest/finding_annotations_bbox_adjusted.csv'
#     dicom_dir = 'C:/Users/piete/Documents/MScThesisLocal/VinDrMammoTest'
#     save_processed_path = 'C:/Users/piete/Documents/MScThesisLocal/VinDrMammoTest/processed_images'
#     # csv_file_path = 'C:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammoOriginal/finding_annotations.csv'
#     # save_csv_path = 'C:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammoOriginal/finding_annotations_bbox_adjusted.csv'
#     # zip_path = 'c:/Users/PCpieter/Documents/vscode/mscthesis/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0.zip'
#     # save_processed_path = 'c:/Users/PCpieter/Documents/vscode/mscthesis/VinDrMammoOriginal/processed_images'

#     find_df = pd.read_csv(csv_file_path)
#     dicom_files = list(scantree(dicom_dir, extention='.dicom'))

#     results = []
#     i = 0
#     for file in dicom_files:
#         result = fit_image(file, find_df, save_processed_path)
#         results.append(result)
#         i += 1
#         if i % 100 == 0:
#             print(f"Processed {i} images")

#     # adjust bounding boxes using vectorized methods
#     bbox_df = save_adjusted_bboxes(results)
#     bbox_df.to_csv('/gpfs/work5/0/prjs0976/physionet.org/files/vindr-mammo/1.0.0/bbox_intermediate_df.csv', index=False)

#     # adjust bounding boxes
#     final_df = adjust_bounding_boxes(find_df, bbox_df)
#     final_df.to_csv(save_csv_path, index=False)

#     end = time.time()
#     print(f"Processing took {end - start}")

def main():
    dicom_data = 'C:/Users/piete/Documents/MScThesisLocal/VinDrMammoTest'

# Execute the main function
if __name__ == "__main__":
    main()
