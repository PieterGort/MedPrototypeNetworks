import torch
import os
from PIL import Image
import argparse
import numpy as np
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def calculate_mean_std(image_folder):
    pixel_values = []
    count = 0
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img) / 255.0  # Scale to [0, 1]
            pixel_values.append(img_array)
        count += 1
        print(f"Processed {count} folders.")

    pixel_values = np.concatenate([x.flatten() for x in pixel_values])
    mean = np.mean(pixel_values)
    std = np.std(pixel_values)

    return mean, std

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)

def preprocess_image(input_folder, output_folder, img_size):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    count = 0

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),  # Convert to 0-255 scale
    ])

    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Construct the path to the input image
            input_path = os.path.join(root, file)

            # Determine the study folder structure relative to the base input folder
            relative_path = os.path.relpath(root, input_folder)
            # Construct the output directory path using the relative path
            output_dir = os.path.join(output_folder, relative_path)

            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Load and transform the image
            img = Image.open(input_path).convert('RGB')
            img_tensor = transform(img).to('cuda')  # Move to GPU

            # Convert back to PIL Image and save
            img_tensor = img_tensor.cpu().byte()  # Move back to CPU and convert to byte
            img_pil = transforms.ToPILImage()(img_tensor)
            output_path = os.path.join(output_dir, file)
            img_pil.save(output_path)
        count += 1
        print(f"Processed {count} folders.")
        if count == 10:
            break

    print("Processing completed.")


def main(args):
    # prepocess_image(args.input_folder, args.output_folder, args.img_size)
    mean, std = calculate_mean_std(args.input_folder)
    print(f'Mean: {mean}, Std: {std}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Resize images in a folder')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder', default='C:/Users/piete/Documents/MScThesisLocal/VinDrMammo/Processed_Images')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder', default='C:/Users/piete/Documents/MScThesisLocal/VinDrMammo/Processed_Images_1536x768')
    parser.add_argument('--img_height', type=int, help='Height of the resized image', default=1536)
    parser.add_argument('--img_width', type=int, help='Width of the resized image', default=768)

    args = parser.parse_args()
    args.img_size = (args.img_height, args.img_width)
    main(args)