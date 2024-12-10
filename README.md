
# 🧘 Yoga Pose Recognition and Real-Time Detection

This project implements a deep learning model for yoga pose recognition using images and real-time video input. It leverages TensorFlow for training the model, OpenCV for real-time pose detection, and image preprocessing for handling input images. The model predicts yoga poses and provides feedback on image quality (e.g., brightness, resolution).

## Features ✨

- **Image Pose Prediction 📸**: Provide an image of a yoga pose, and the model will predict the pose and provide a confidence score.
- **Real-time Pose Detection 🎥**: Detect yoga poses through live webcam video feed and display the pose name with a confidence score.
- **Image Quality Feedback 🖼️**: Analyze the input image for resolution and brightness issues, offering suggestions for improvements.
- **Data Augmentation 🔄**: Apply augmentation techniques like shear, zoom, and horizontal flip to enhance the training dataset.

## Prerequisites ⚙️

Make sure the following libraries are installed:

- `tensorflow`
- `opencv-python`
- `numpy`
- `matplotlib`
- `pillow`

You can install them using the following command:

```bash
pip install tensorflow opencv-python numpy matplotlib pillow
```

## Dataset 📂

This model requires a yoga pose dataset. The dataset should be in a folder structure with subdirectories for each class (yoga pose). The dataset must be extracted from a ZIP file, and the images inside must be organized by class for training.

## Setup and Training ⚡

### Step 1: Dataset Extraction 📦

Place the ZIP file containing the dataset in the specified path (`C:\Users\HP\Downloads\archive (1).zip`). The dataset will be extracted to a folder named `yoga_poses`.

### Step 2: Model Training 🧠

The model is trained using a Convolutional Neural Network (CNN) for image classification. It performs image classification using a Convolutional Neural Network (CNN) architecture. Data augmentation techniques are applied during the training process to improve the model's robustness.

Run the training code to start the training process and save the model weights after completion.

### Step 3: Model Weights 💾

The trained model's weights are saved to `model_weights.weights.h5`. You can use this file for predictions without retraining the model.

## Usage 🚀

### Option 1: Provide an Image File 🖼️

1. Start the program and choose option 1.
2. Provide the path to the image file you want to classify.
3. The program will output the predicted pose, confidence score, and any required corrections.

### Option 2: Real-time Pose Detection 🎥

1. Start the program and choose option 2.
2. The program will begin detecting yoga poses through your webcam in real-time and display the predicted pose along with confidence.

### Option 3: Exit ❌

Select this option to exit the program.

## Conclusion 🎉

This project helps in recognizing yoga poses using deep learning and real-time video input. It can be extended further to improve its accuracy, support more poses, or implement advanced image quality corrections.
