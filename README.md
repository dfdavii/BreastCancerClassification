I read the "Vesal, S., Ravikumar, N., Davari, A., Ellmann, S., & Maier, A.K. (2018). Classification of breast cancer histology images using transfer learning. ArXiv, abs/1802.09424."

This is my attempt of regenerationg the code. I wrote the classes: ModifiedInceptionV3 and ModifiedResNet50. One of the class has been commented out. You can use 'image' or 'patch' as whichever mode you want to run. The epoch number is 100 according to the paper. It takes an extremely long time to train, but the training and val accuracies get better, and the losses get lower at each epoch. Fell free to run it.



# Bach2018Dataset Class Explanation

The `Bach2018Dataset` class is a custom dataset class for handling histopathology images. It includes methods for loading data, preprocessing images, and applying data augmentation techniques. Below is a detailed explanation of each part of the code:

## Reinhard Normalization Function

The `reinhard_normalization` function applies Reinhard stain normalization to the source image using the target image. This process ensures that the color distribution of the source image matches that of the target image, which is particularly useful in histopathology to standardize the appearance of images.

### Steps Involved:

1. **Convert Images to LAB Color Space**:
   - The function first converts both the source and target images from RGB to LAB color space. The LAB color space separates the luminance (L) from the color information (A and B), making it easier to manipulate colors.

2. **Calculate Mean and Standard Deviation**:
   - It calculates the mean and standard deviation for each channel (L, A, B) of both the source and target images.

3. **Normalize Each Channel**:
   - The function normalizes each channel of the source image by subtracting the source mean and dividing by the source standard deviation. It then scales the normalized values by the target standard deviation and adds the target mean. This step adjusts the color distribution of the source image to match that of the target image.

4. **Convert Back to RGB Color Space**:
   - Finally, the function converts the normalized LAB image back to RGB color space. The resulting image has the same color distribution as the target image.

### Summary:

The `reinhard_normalization` function is essential for standardizing the appearance of histopathology images, making it easier to compare and analyze images from different sources. By matching the color distribution of the source image to that of the target image, this function helps in reducing variability and improving the consistency of image analysis.

## Bach2018Dataset Class

### Initialization

The `__init__` method initializes the dataset. It takes several parameters, including the root directory of the dataset, the path to the target image for normalization, the data split (train, val, test), the transformation to apply, the patch size, the random seed, and the mode (patch or whole image).

### Loading Data

The `_load_data` method loads the data from the specified directory. It shuffles the images and splits them into training, validation, and test sets based on the provided split parameter. The method returns a DataFrame containing the filenames and their corresponding classes.

### Length of Dataset

The `__len__` method returns the length of the dataset.

### Getting an Item

The `__getitem__` method retrieves an image and its label based on the provided index. It loads the image, applies Reinhard normalization, and processes the image based on the specified mode (patch or whole image). If the mode is 'patch', the image is divided into patches with 50% overlap, and data augmentation is applied. If the mode is 'whole image', the image is resized and transformed.

### Getting the Label

The `get_label` method returns the numerical label corresponding to the class name.

## Example Usage

An example shows how to create an instance of the `Bach2018Dataset` class with the specified parameters.

# Trainer Class Explanation

The `Trainer` class is designed to facilitate the training, validation, and testing of deep learning models using PyTorch. It supports both traditional models and models that require patch-based processing of images.

## Key Components

### 1. **Class Initialization (`__init__` method)**
The `__init__` method initializes the trainer class with the following parameters:
- **model**: The neural network model to be trained.
- **train_loader**: The DataLoader for the training data.
- **val_loader**: The DataLoader for the validation data.
- **test_loader**: Optional, the DataLoader for the test data.
- **num_epochs**: The number of epochs for training (default is 100).
- **lr**: Learning rate for the optimizer (default is `1e-4`).
- **mode**: The mode of training, either 'patch' (for patch-based input) or not.

It also initializes lists to store losses and accuracies for both training and validation over epochs.

### 2. **Training Method (`train` method)**
This method orchestrates the training process:
- It calls the `train_fold` method to train the model on the given data.
- After training, it prints the validation metrics and plots the loss and accuracy graphs over epochs using the `plot_metrics` method.

### 3. **Training for Each Fold (`train_fold` method)**
The `train_fold` method handles the core training loop:
- Sets the criterion (loss function) as Cross-Entropy Loss.
- Uses the SGD optimizer with momentum and Nesterov acceleration.
- Transfers the model to the GPU (if available).
- Iterates through the specified number of epochs, training the model with the `train_epoch` method and validating using the `validate` method.

It tracks and stores losses and accuracies for each epoch and calculates final validation metrics using the `calculate_metrics` method.

### 4. **Training Per Epoch (`train_epoch` method)**
This method handles training for one epoch:
- Sets the model to training mode.
- Loops over batches of data, moves them to the device, and processes them based on the `mode` (either standard or patch-based).
- For each batch, it computes the loss, performs backpropagation, and updates the model's weights.
- It calculates and prints the average training loss and accuracy for the epoch.

### 5. **Validation (`validate` method)**
The `validate` method evaluates the model's performance on the validation set:
- It sets the model to evaluation mode (disabling gradients).
- Loops over batches of validation data, computes the outputs, and calculates the validation loss.
- Tracks the predicted labels and ground truth labels to calculate accuracy.
- Returns the average validation loss and accuracy, along with predicted and true labels for metric calculation.

### 6. **Metrics Calculation (`calculate_metrics` method)**
This method calculates various performance metrics for the model:
- **Confusion Matrix**: Provides a matrix representing correct and incorrect classifications.
- **Accuracy**: The proportion of correctly predicted samples.
- **Recall**: The ability of the model to identify positive cases.
- **Precision**: The ability of the model to avoid false positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the ROC curve (measures classifier performance).

### 7. **Prediction (`predict` method)**
The `predict` method is used to generate predictions for a given data loader:
- Runs the model in evaluation mode and processes the input data, either using patches or regular images.
- Outputs the softmax probabilities for each class.

### 8. **Plotting Metrics (`plot_metrics` method)**
This method generates plots for training and validation loss and accuracy over epochs:
- Uses Matplotlib to create two subplots: one for loss and one for accuracy.
- Displays the graphs to visualize model performance over the training period.

## Key Features

- **Patch-Based Mode**: The class can handle training on patch-based data, which is common in certain image-based models.
- **Metrics Calculation**: Automatically computes important metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
- **Visualization**: Plots loss and accuracy over epochs for both training and validation sets, giving insights into model performance.
- **GPU Support**: The class can automatically run on CUDA if available.

## Usage Example

To use the `Trainer` class:
1. Initialize it with your model, training, validation, and optional test loaders.
2. Call the `train()` method to train the model.
3. After training, metrics will be printed, and plots will be generated.

```python
trainer = Trainer(model, train_loader, val_loader, num_epochs=50, lr=1e-3)
trained_model, validation_metrics = trainer.train()


