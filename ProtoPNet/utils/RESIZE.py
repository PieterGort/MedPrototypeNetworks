import os
from PIL import Image
import argparse

def prepocess_image(input_folder, output_folder, img_size):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        count = 0
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

            # Load the image
            img = Image.open(input_path)
            # Resize the image
            img = img.resize(img_size)
            
            # convert to RGB
            img = img.convert('RGB')

            # Construct the path to the output image
            output_path = os.path.join(output_dir, file)
            # Save the resized image
            img.save(output_path)
            count += 1
            print(f"Resized {count} folders.")

    print("Resizing completed.")

def main(args):
    prepocess_image(args.input_folder, args.output_folder, args.img_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize images in a folder')
    parser.add_argument('input_folder', type=str, help='Path to the input folder')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')
    parser.add_argument('img_height', type=int, help='Heigth of the resized image')
    parser.add_argument('img_width', type=int, help='Width of the resized image')
    args = parser.parse_args()
    args.img_size = (args.img_height, args.img_width)
    main(args)