# fire-prediction-using-image
# Overview

The Fire Prediction Using Image project is a machine learning application that identifies the presence of forest fires using
images. This project uses a deep learning model built with TensorFlow and Keras to detect whether an image depicts a wildfire or
not. The model is trained using a dataset of images, with each image labeled as either showing a wildfire or not. The project uses
a Convolutional Neural Network (CNN) to classify these images, leveraging image preprocessing techniques for optimal accuracy.

The primary goal of this project is to aid in the early detection of wildfires, which is critical for timely responses to
mitigate potential damage to forests and human settlements.

# Key Features

Training and Validation: The model is trained on a large dataset consisting of images categorized into two classes: "Wildfire"
and "No Wildfire." Image data is preprocessed using rescaling techniques to normalize the pixel values for training.
Model Architecture: The CNN model consists of multiple convolutional and pooling layers to extract image features, followed by
fully connected layers that output a classification prediction (wildfire or no wildfire).
GUI Interface: A simple graphical user interface (GUI) built with Tkinter allows users to upload images. The uploaded image is
processed, and the model predicts whether the image contains a wildfire or not.
Real-time Prediction: The model processes new images using a trained model, displaying the prediction result ("Wildfire" or "No
Wildfire") in real-time.

# Project Structure

Training and Testing:
Images are organized in the training, validation, and test directories for proper dataset handling.
Image data is augmented using ImageDataGenerator to improve the generalization of the model.
Deep Learning Model:
The core of the project is a CNN model that includes layers for convolution, pooling, and fully connected layers.
Prediction:
The trained model can predict on new images that are uploaded via the GUI.
The image is preprocessed to match the format expected by the model (resized to 64x64 pixels and normalized).

# Installation

To run this project, you'll need to install the following dependencies:

TensorFlow: Used for model creation and training.
Keras: Used for building the neural network.
Tkinter: Used for the graphical user interface.
PIL: Used for image handling

# How It Works

Data Preprocessing: The images are preprocessed to a consistent size (64x64 pixels) and rescaled to ensure consistent model input.
Model Training: The CNN model is compiled using the Adam optimizer and binary cross-entropy loss, ideal for binary classification
tasks. The model is then trained over multiple epochs to optimize its accuracy.
Image Upload & Prediction: Once the model is trained, users can upload new images via the GUI. The system will then predict
whether the image contains a wildfire or not, and the result will be displayed on the screen.

# How to use
Run the script to start the GUI application.
Click the "Upload Image" button to select an image.
The application will display the image and show the prediction ("Wildfire" or "No Wildfire").

# Conclusion

This project demonstrates the use of deep learning and computer vision techniques for the important task of wildfire detection.
The implementation of a simple graphical interface makes it easy to use, even for individuals with no coding experience.

