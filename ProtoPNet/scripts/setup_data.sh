#!/bin/bash

# Navigate to the target directory where the data directory will be created
cd C:/Users/piete/Documents/MScThesisLocal/data
# Create the data directory and manage downloads within it
mkdir -p data
cd data

# Continue with your existing script
for input_size in 224
do
    kaggle datasets download -d cdeotte/jpeg-isic2019-${input_size}x${input_size}
    kaggle datasets download -d cdeotte/jpeg-melanoma-${input_size}x${input_size}
    unzip -q jpeg-melanoma-${input_size}x${input_size}.zip -d jpeg-melanoma-${input_size}x${input_size}
    unzip -q jpeg-isic2019-${input_size}x${input_size}.zip -d jpeg-isic2019-${input_size}x${input_size}
    rm jpeg-melanoma-${input_size}x${input_size}.zip jpeg-isic2019-${input_size}x${input_size}.zip
done
