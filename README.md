# Fish Classification Project

This project is a deep learning-based fish classification study using the Kaggle Fish Dataset. Fish species are classified using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The model is designed for multi-class classification and includes preprocessing, training, and evaluation steps.

## Project Overview

The aim of this project is to classify fish species based on visual data. The model uses a simple ANN architecture built with TensorFlow's Keras library's Sequential API. The output layer uses the `softmax` activation function, which returns class probabilities for fish species.

## Dataset

The dataset used is the Kaggle Fish Dataset, which contains images of various fish species. The data is loaded and preprocessed using TensorFlow's `ImageDataGenerator` class.

## Model Architecture

- **Input Layer**: 
  - A `Flatten` layer is used to convert the input images of size 64x64x3 into a one-dimensional array.
- **Hidden Layers**: 
  - Three fully connected `Dense` layers with ReLU activation function were added:
    - Layer 1: 512 units
    - Layer 2: 256 units
    - Layer 3: 128 units
- **Output Layer**: 
  - A `Dense` layer with `softmax` activation function for multi-class classification was added. The number of neurons in the output layer is dynamically adjusted based on the number of classes in the dataset.

## Model Compilation

The model was compiled with the following settings:
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy (as it is a multi-class classification)
- **Evaluation Metrics**: Accuracy

## Training

The model was trained for 10 epochs with a batch size of 128. The training and validation sets were used to monitor the model's performance.

## Results

The accuracy on the test set was calculated as 0.9222222222222223. Performance can be further improved by tuning hyperparameters such as the number of layers, the number of units in the layers, learning rate, and dropout rates. Visualizations of the training process and fish species classification can be found in the notebook.

## Kaggle Link
https://www.kaggle.com/code/derrrd/fish-classification-w-ann


