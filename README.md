🧘‍♂️ Yoga Pose Prediction System
This is a Yoga Pose Prediction System that leverages TensorFlow and Keras to identify yoga poses in images and video streams. The system predicts the yoga pose from a given image and suggests corrections if the image quality is suboptimal.

📂 Table of Contents
Features
Installation
Usage
Model Architecture
Training Process
Real-time Pose Detection
Image Quality Analysis
Contributions
🚀 Features
Yoga Pose Prediction: Classify yoga poses from a provided image.
Real-Time Pose Detection: Use your webcam for real-time yoga pose detection.
Image Quality Analysis: Get feedback on image resolution and brightness.
Corrections: Receive suggestions to improve image quality for better pose prediction.
Customizable: You can easily integrate with other pose datasets for further training or adaptation.
💻 Installation
Prerequisites
Python 3.x: The code is written for Python 3.
Libraries: Install the necessary dependencies using pip:
bash
Copy code
pip install tensorflow opencv-python Pillow matplotlib
Dataset: Download the yoga poses dataset and unzip it to the yoga_poses folder.
🏃‍♂️ Usage
Run the Code: After installation, you can start the program by executing:
bash
Copy code
python yoga_pose_prediction.py
Interactive Mode: Once the program runs, you'll be prompted with a menu:
(1) Provide an image file: Input a local image path to predict the yoga pose.
(2) Real-time pose detection: Start the webcam to detect poses in real-time.
(3) Exit: To exit the application.
Provide an Image
Enter the full path to an image when prompted.
The program will predict the yoga pose and, if necessary, provide corrections (e.g., image resolution too low or brightness too high).
Real-Time Pose Detection
You can start real-time yoga pose detection using your webcam.
The model will display the predicted pose and any image quality corrections directly on the video feed.
🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) designed for classifying yoga poses from images. The architecture is as follows:

Convolution Layers: Extract features from the input image using filters.
Max-Pooling Layers: Down-sample the image dimensions to retain important features.
Fully Connected Layers: Learn non-linear combinations of features to make the final classification.
Output Layer: Uses softmax activation to predict the yoga pose class.
⚙️ Training Process
Dataset Preprocessing:

Images are resized to 128x128 pixels.
Data augmentation techniques (shearing, zoom, flipping) are applied to improve model generalization.
Model Training:

The model is trained using the Adam optimizer and categorical cross-entropy loss.
Epochs: The model is trained for 10 epochs, but you can adjust this for better performance.
Saving the Model:

After training, the model weights are saved as model_weights.weights.h5.
🎥 Real-time Pose Detection
The real-time pose detection leverages OpenCV for capturing webcam video frames.
The frames are processed to predict the yoga pose using the trained model.
Corrections (resolution, brightness) are displayed on the screen to help improve image quality.
🖼️ Image Quality Analysis
The system checks the quality of input images before prediction:

Resolution: Ensures the image is at least 128x128 pixels.
Brightness: Analyzes the average brightness and suggests corrections if it's too dark or too bright.
🤖 Contributions
Feel free to contribute to the project! Whether it's improving the model, adding new features, or fixing bugs, your contributions are welcome.

Example Output:
Image Path Input:

yaml
Copy code
Enter the path to the image file: "path/to/your/image.jpg"
Predicted Pose: Warrior Pose, Confidence: 85.56%
Corrections Needed:
- Image is too dark. Increase brightness.
Real-time Pose Detection:

sql
Copy code
Starting real-time pose detection...
Pose: Warrior Pose (85.56%)
Image is too dark. Increase brightness.
Enjoy your yoga pose detection experience! 🧘‍♀️💪
