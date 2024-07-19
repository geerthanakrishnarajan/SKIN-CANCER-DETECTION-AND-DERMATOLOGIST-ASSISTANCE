# AI-Powered Skin Cancer Detection and Dermatologist Consultation Assistance

## Abstract
This project integrates deep learning algorithms, specifically Convolutional Neural Networks (CNNs), for precise skin lesion classification and staging, facilitating early skin cancer detection. Additionally, it includes a dermatologist consultation assistant powered by Natural Language Processing (NLP), offering personalized advice and promoting skin health awareness.

## Modules

### 1. Image Data Preprocessing
- Resize, standardize, and normalize input images.
- Augment images to diversify the dataset and address class imbalances.

### 2. CNN Model Training
- Design and train a CNN architecture on a split dataset.
- Optimize with techniques like batch normalization and dropout layers.

### 3. Cancer Type and Staging Integration
- Classify malignancies based on skin lesion characteristics.
- Integrate staging information into the detection model.

### 4. User Interface (UI) - Webpage
- Develop a web or mobile app for users to upload images and receive analysis.
- Display model predictions with confidence scores.

### 5. Dermatologist Chatbot Application
- Provide personalized treatment suggestions and educational content.
- Streamline appointment scheduling with dermatologists.

### 6. Dermatologist Consultation Scheduling and Referral
- Integrate with hospital websites for easy appointment booking and referrals.

### 7. Skin Cancer Awareness and Education
- Develop an educational module focused on skin cancer prevention and detection.

## Technologies Used
- Convolutional Neural Networks (CNNs)
- Natural Language Processing (NLP)
- OpenCV for image processing
- Keras for deep learning
- TensorFlow
- Flask

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/geerthanakrishnarajan/SKIN-CANCER-DETECTION-AND-DERMATOLOGIST-ASSISTANCE.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

FOR DATASET:
https://www.kaggle.com/datasets/geerthanak/7-types-of-skin-cancer/data

Usage
1. Run the preprocessing script to prepare the images.
2. Train the CNN model with your dataset.
3. Deploy the web application for user interaction.
4. Use the chatbot for personalized dermatological advice.
