# Brain Tumor MRI Classification

This repository contains code for training a Convolutional Neural Network (CNN) to classify brain tumor MRI images using the EfficientNetB7 architecture. The dataset used for this project is the Brain Tumor MRI Dataset, which contains MRI images categorized into four classes: glioma, meningioma, pituitary, and no tumor.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to develop a CNN model that can accurately classify MRI images of brain tumors into one of four categories. The EfficientNetB7 architecture is used as the base model, which is fine-tuned on the Brain Tumor MRI Dataset. The project includes data preprocessing, model training, evaluation, and prediction on new images.

## Dataset

The Brain Tumor MRI Dataset can be found on Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset contains two main folders:
- `Training`: Contains training images
- `Testing`: Contains testing images

Each folder has subfolders corresponding to the four classes of brain tumors.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/brain-tumor-mri-classification.git
    cd brain-tumor-mri-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the dataset downloaded and extracted into the appropriate directories.

2. Run the training script:
    ```bash
    python train.py
    ```

3. The model will be saved as `final_model.h5` in the working directory.

## Model Architecture

The model uses EfficientNetB7 as the base model, followed by:
- GlobalAveragePooling2D layer
- Dropout layer
- Dense layer with 512 units and ReLU activation
- Dropout layer
- Dense layer with 4 units and softmax activation

## Training

The training script includes data augmentation techniques such as rotation, width/height shift, shear, zoom, and horizontal flip. Early stopping and model checkpointing are used to save the best model based on validation accuracy.

## Evaluation

After training, the model is evaluated on the test set. The script prints the test loss and accuracy.
![image](https://github.com/user-attachments/assets/2733a657-efd0-4fe8-8f23-2d1c9143614e)


## Prediction

You can use the trained model to predict the class of a new MRI image. An example usage is provided in the script:
```python
image_path = '/path/to/your/image.jpg'
predicted_class = predict_image(image_path, model, img_size, class_labels)
print(f'Predicted class: {predicted_class}')
```

## Results

The results of the training and validation process are visualized through accuracy and loss plots. The final test accuracy is also printed.

## Acknowledgements

- The dataset used in this project is from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- The EfficientNetB7 model is provided by TensorFlow Keras.

