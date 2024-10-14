import os
from PIL import Image
import numpy as np

def update_mean_std(count, mean, M2, new_data):
    count += new_data.shape[0]
    delta = new_data - mean
    mean += delta.sum(axis=0) / count
    delta2 = new_data - mean
    M2 += (delta * delta2).sum(axis=0)
    return count, mean, M2

def finalize_mean_std(count, mean, M2):
    variance = M2 / count
    std_dev = np.sqrt(variance)
    return mean, std_dev

def calculate_mean_std_incrementally(directory):
    print("Starting to calculate mean and standard deviation incrementally.")
    count = 0
    mean = None
    M2 = None

    # Loop over all files in the directory
    print("Reading images from", directory)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            with Image.open(filepath) as img:
                img = img.resize((224, 224))  # Ensure image is 224x224
                img_array = np.array(img, dtype=np.float32)
                if img_array.ndim == 3 and img_array.shape[2] == 3:  # Ensure it is a color image
                    if mean is None:  # Initialize mean and M2 arrays
                        mean = np.zeros_like(img_array.mean(axis=(0, 1)), dtype=np.float64)
                        M2 = np.zeros_like(mean)
                    img_array = img_array.reshape(-1, 3)  # Flatten spatial dimensions but keep channels
                    count, mean, M2 = update_mean_std(count, mean, M2, img_array)
                    print("Processed", count, "images, updated mean and std. are", mean, M2)

    
    if count > 0:
        mean, std = finalize_mean_std(count, mean, M2)
        return mean, std
    else:
        return None, None
    
    
# Usage
directory = os.getenv('ISIC_IMAGE_FOLDER', "/gpfs/work5/0/prjs0976/ISIC_2020_Training_JPEG_224_224/train")
mean, std = calculate_mean_std_incrementally(directory)
if mean is not None and std is not None:
    print("Mean:", mean)
    print("Standard Deviation:", std)
else:
    print("No images found or images are not in expected format.")