# Violence-Detection-using-Transfer-Learning
This deep learning project uses transfer learning four pre-trained CNN models to detct violence in a video. InceptionV3, MobileNetV2, ResNet50V2, and VGG19 are the models used to classify a video contains violence or not and perform a comparison between the models
* Overview
* Key Highlights
* Models 
* Performance Comparison
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

#  Performance Comparison
--------------------------------------------------------------------------------------------------------------------------------
| Rank | Model          | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) | Correct Predictions | Wrong Predictions |
|------|--------------- |----------|-----------------|--------------|----------------|---------------------|-------------------|
| 1    | MobileNetV2    | 94%      | 94%             | 94%          | 94%            | 2196 / 2340         | 144               |
| 2    | ResNet50V2     | 93%      | 93%             | 93%          | 93%            | 2187 / 2340         | 153               |
| 3    | InceptionV3    | 92%      | 92%             | 92%          | 92%            | 2154 / 2340         | 186               |
| 4    | VGG19          | 74%      | 76%             | 73%          | 72%            | 1726 / 2340         | 614               |
--------------------------------------------------------------------------------------------------------------------------------
# Dataset
The dataset is made up of YouTube videos with a range of settings that depict 
both violent and non-violent human behaviour. The violent videos, which were 
shot in a variety of settings with varying camera angles and lighting conditions, 
feature actual street fights and other hostile behaviours.The dataset was initially made up of 1000 violent and 1000 non-violent videos, 
with a wide variety of sequences to guarantee a thorough depiction of current 
situations. However, in order to address computational issues and the focus of 
this study, the dataset was shrunk to a more manageable size.

For more details and to access the dataset, please visit the following link:  
https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence situations-dataset 
