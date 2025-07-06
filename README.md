# Main_Code.ipynb

This Jupyter Notebook, `Main_Code.ipynb`, is designed for image processing and dataset preparation, primarily focusing on medical image analysis, specifically for the Kvasir-Capsule dataset. It includes steps for handling compressed image archives, metadata processing, and setting up a custom dataset for machine learning tasks, likely for classification or object detection of gastrointestinal findings.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Key Components](#key-components)
- [Transformations](#transformations)

## Project Overview
The notebook automates the extraction of images from `.tar.gz` archives, cleans and processes associated metadata, and constructs a PyTorch-compatible dataset. It specifically filters out anatomical data, focusing on disease classification. This setup is crucial for training deep learning models on the Kvasir-Capsule dataset.

## Features
- **Archive Extraction**: Automatically unzips and extracts `.tar.gz` image archives.
- **Metadata Processing**: Reads and processes a CSV metadata file, extracting bounding box coordinates and disease classifications.
- **Data Cleaning**: Removes irrelevant anatomical data from the metadata to focus on pathological findings.
- **Custom Dataset**: Implements a `KvasirDataset` class for efficient loading and preprocessing of images and their corresponding labels and bounding boxes.
- **Data Splitting**: Splits the processed data into training and testing sets.
- **Image Transformations**: Defines Albumentations-based transformations for data augmentation and preprocessing, including resizing, normalization, and data augmentation techniques like horizontal flips, brightness/contrast adjustments, and shift/scale/rotate operations.

## Dataset
The notebook is designed to work with the Kvasir-Capsule dataset, which should be stored in your Google Drive. Specifically, it expects `.tar.gz` image files and an associated `metadata.csv` file within the `/content/drive/My Drive/kvasir-capsule/labelled_images` directory.

## Prerequisites
- Python 3.x
- Google Colab environment (recommended for seamless Google Drive integration and GPU access)

## Installation
The notebook handles most of the installations by running `!pip install albumentations`. Other required libraries are standard and usually pre-installed in Colab or easily installable via pip:
- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python-headless` (cv2)
- `Pillow` (PIL)
- `torch`
- `torchvision`
- `scikit-learn`
- `PyYAML`
- `pydantic`

## Usage
1.  **Mount Google Drive**: The notebook starts by mounting Google Drive to access the dataset. Ensure your Kvasir-Capsule dataset is placed in `/content/drive/My Drive/kvasir-capsule/labelled_images`.
2.  **Extract Data**: Run the cells to unzip and extract image files from the `.tar.gz` archives into the `/content/unzipped` directory.
3.  **Process Metadata**: The metadata CSV is loaded, cleaned, and enhanced with bounding box information. A new CSV `processed_kvasir_data.csv` is generated.
4.  **Create Datasets**: The `KvasirDataset` class is instantiated using the processed data, and the data is split into `train_kvasir_data.csv` and `test_kvasir_data.csv`.
5.  **Apply Transforms**: Image transformations for training and testing are defined using `albumentations`.
6.  **Data Exploration**: The `plot_sample_with_box` method can be used to visualize images with their bounding boxes and labels for verification.

## File Structure
- `Final_Code.ipynb`: The main Jupyter notebook.
- `/content/drive/My Drive/kvasir-capsule/labelled_images/`: Expected directory for raw `.tar.gz` image archives and `metadata.csv`.
- `/content/unzipped/`: Temporary directory where images are extracted.
- `metadata.csv`: Original metadata file (expected in Google Drive).
- `new_meta.csv`: Intermediate metadata file after initial cleaning.
- `processed_kvasir_data.csv`: Final processed metadata with bounding box information.
- `train_kvasir_data.csv`: CSV file containing paths and labels for training images.
- `test_kvasir_data.csv`: CSV file containing paths and labels for testing images.

## Key Components

### `KvasirDataset` Class
A custom PyTorch `Dataset` that:
- Loads image paths and labels from a CSV file.
- Opens and converts images to RGB format.
- Retrieves and normalizes bounding box coordinates for diseased cases.
- Applies specified image transformations using `albumentations`.
- Provides a `plot_sample_with_box` method to visualize data samples with bounding boxes.

### Data Preprocessing Steps
- Unzipping and extracting `.tar.gz` files containing images.
- Removing anatomical categories from the metadata.
- Calculating `x_min`, `y_min`, `x_max`, `y_max` for bounding boxes based on provided polygon coordinates.
- Merging image file information with metadata.

## Transformations

The notebook defines three sets of transformations using the `albumentations` library:

-   **`get_train_transform()`**:
    -   Resizes images to 224x224 pixels.
    -   Applies `HorizontalFlip` with p=0.5.
    -   Applies `RandomBrightnessContrast` with p=0.2.
    -   Applies `ShiftScaleRotate` with shift\_limit=0.05, scale\_limit=0.05, rotate\_limit=15, p=0.5.
    -   Normalizes pixel values.
    -   Converts the image to a PyTorch tensor (`ToTensorV2`).
    -   Configured for bounding box parameters in `pascal_voc` format.

-   **`get_val_transform()`**:
    -   Resizes images to 224x224 pixels.
    -   Normalizes pixel values.
    -   Converts the image to a PyTorch tensor (`ToTensorV2`).
    -   Configured for bounding box parameters in `pascal_voc` format.

-   **`get_test_transform()`**:
    -   Identical to `get_val_transform()`, resizing and normalizing images for testing.
    -   Resizes images to 224x224 pixels.
    -   Normalizes pixel values.
    -   Converts the image to a PyTorch tensor (`ToTensorV2`).
    -   Configured for bounding box parameters in `pascal_voc` format.

This README provides a comprehensive overview for anyone looking to understand, use, or extend the `Final_Code.ipynb` notebook.
