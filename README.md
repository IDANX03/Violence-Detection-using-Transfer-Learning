# Violence-Detection-using-Transfer-Learning
This deep learning model utilises transfer learning with four pre-trained CNN models to classify video frames as violent or non-violent.
* Overview
* Key Highlights
* Models Used
* Performance Comparison
* Dataset
* Technologies
* Project Structure
* Installation
* Usage
* Methodology
* Results

# 🎯 Overview
This project implements a violence detection system that uses transfer learning with pre-trained Convolutional Neural Networks (CNNs). The model analyses video frames and classifies them as violent or non-violent. By utilising transfer learning, the project achieves high accuracy while reducing training time and computational requirements. MobileNetV2 is considered to be the best performing model among other models with accuracy up to 94%.

# ✨ Key Hightlights
* Best Model: MobileNetV2 achieving 94% accuracy.
* Four Model Comparison: Comprehensive analysis across different architectures.
* Detailed Metrics: Precision, Recall, F1-Score, and Confusion Matrices.
* Advanced Augmentation: Multiple techniques for robust model training.
* Efficient Training: Transfer learning reduces training time significantly.
* Stratified Splitting: Ensures balanced class distribution.
* Production-Ready: Complete pipeline from video to predictions.

# Models Implemented
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

# 📊 Performance Comparison
