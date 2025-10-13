# Violence-Detection-using-Transfer-Learning
This deep learning project uses transfer learning four pre-trained CNN models to detct violence in a video. InceptionV3, MobileNetV2, ResNet50V2, and VGG19 are the models used to classify a video contains violence or not and perform a comparison between the models
* Overview
* Key Highlights
* Models 
* Dataset
* Methodology
* Results

#  Overview
This project implements a violence detection system that uses transfer learning with pre-trained Convolutional Neural Networks (CNNs). The model analyses video frames and classifies them as violent or non-violent. By utilising transfer learning, the project achieves high accuracy while reducing training time and computational requirements. MobileNetV2 is considered to be the best performing model among other models with accuracy up to 94%.

#  Key Hightlights
* Best Model: MobileNetV2 achieving 94% accuracy.
* Four Model Comparison: Comprehensive analysis across different architectures.
* Detailed Metrics: Precision, Recall, F1-Score, and Confusion Matrices.
* Advanced Augmentation: Multiple techniques for robust model training.
* Efficient Training: Transfer learning reduces training time significantly.
* Stratified Splitting: Ensures balanced class distribution.
* Production-Ready: Complete pipeline from video to predictions.

# Models 
1. MobileNetV2
   * Accuracy: 94%
   * Architecture: Lightweight, efficient for mobile/edge deployment.
   * Parameters: ~3.4M trainable.
   * Best For: Resource-constrained environments, real-time applications.
2. ResNet50V2
   * Accuracy: 93%
   * Architecture: Deep residual learning with skip connections.
   * Parameters: ~23.5M trainable.
   * Best For: High accuracy with moderate computational cost.
3. InceptionV3
   * Accuracy: 92%
   * Architecture: Multi-scale feature extraction with inception modules.
   * Parameters: ~21.8M trainable.
   * Best For: Balanced performance and feature diversity.
4. VGG19
   * Accuracy: 74%
   * Architecture: Deep sequential architecture with small filters.
   * Parameters: ~20M trainable.
   * Best For: Feature extraction, transfer learning baseline.

# Dataset
The dataset is made up of YouTube videos with a range of settings that depict 
both violent and non-violent human behaviour. The violent videos, which were 
shot in a variety of settings with varying camera angles and lighting conditions, 
feature actual street fights and other hostile behaviours.The dataset was initially made up of 1000 violent and 1000 non-violent videos, 
with a wide variety of sequences to guarantee a thorough depiction of current 
situations. However, in order to address computational issues and the focus of 
this study, the dataset was shrunk to a more manageable size.

For more details and to access the dataset, please visit the following link:  
https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset 

# Methodology

## 1. Video Frame Extraction Pipeline
Video Input → Frame Extraction → Augmentation → Preprocessing → Model Input
### Frame Extraction Process:
* Open video using OpenCV.
* Extract every 7th frame to reduce redundancy.
* Apply data augmentation techniques.
* Convert BGR to RGB color space.
* Resize to 128×128 pixels.
* Normalize pixel values to [0, 1].

### Data Augmentation Techniques:
* Horizontal Flip: Simulates variations in orientation.
* Zooming: scaling the picture by 30%.
* Brightness: Randomly changes brightness between 100% and 130%.
* Random Rotation: -25° to +25° range.

## 2. Training 
The dataset is split using stratified sampling to maintain balanced class distribution, with 70% allocated for training, 15% for validation, and 15% for testing. The optimisation techniques have been used for all the models to optimise 
training and improve performance. Thses techniques help in preventing 
overfitting, reduce training time and enhance the model’s generalisation. 
### Loss Function & Optimizer
* Loss:Binary Crossentropy
* Optimizer: Adam (adaptive learning rate)
### Regularization Techniques
* L2 Regularization: Has been applied to dense layers which prevents overfitting by penalising large weights. Regularization strength: λ = 0.0001.
* Dropout: Reduces co-adaptation of neurons.
* Batch Normalization: Allows higher learning rates.
### Callbacks & Training Control
* ModelCheckpoint: Saves best weights based on validation loss. Weights saved in .h5 format.
* EarlyStopping: Monitors validation loss. Restores best weights automatically
* ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.
* Custom Callback: Stops training when accuracy ≥ 99.9%.
  
# Results 
--------------------------------------------------------------------------------------------------------------------------------
| Rank | Model          | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) | Correct Predictions | Wrong Predictions |
|------|--------------- |----------|-----------------|--------------|----------------|---------------------|-------------------|
| 1    | MobileNetV2    | 94%      | 94%             | 94%          | 94%            | 2196 / 2340         | 144               |
| 2    | ResNet50V2     | 93%      | 93%             | 93%          | 93%            | 2187 / 2340         | 153               |
| 3    | InceptionV3    | 92%      | 92%             | 92%          | 92%            | 2154 / 2340         | 186               |
| 4    | VGG19          | 74%      | 76%             | 73%          | 72%            | 1726 / 2340         | 614               |
--------------------------------------------------------------------------------------------------------------------------------
MobileNetV2 emerged as the top performer with 94% accuracy, demonstrating that modern efficient architectures can outperform larger models. ResNet50V2 followed closely at 93%, leveraging its deep residual connections for strong feature learning. InceptionV3 achieved 92% with its multi-scale feature extraction approach. VGG19 significantly lagged at 74%, highlighting how older sequential architectures struggle compared to modern designs with skip connections and efficient convolutions.
