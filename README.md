# Dog Vision - Dog Breed Image Classifier

A deep learning project that uses computer vision to identify dog breeds from images using TensorFlow and transfer learning.

## 🐕 Overview

This project implements an end-to-end multiclass image classifier capable of identifying 120 different dog breeds from photographs. Built using TensorFlow and Keras with transfer learning techniques, the model achieves 68% accuracy across all breed categories.

## 🎯 Features

- **Multi-class Classification**: Identifies 120 different dog breeds
- **Transfer Learning**: Leverages pre-trained MobileNetV2 architecture for efficient training
- **Data Augmentation**: Implements advanced preprocessing techniques for improved model robustness
- **Performance Optimization**: Uses TensorBoard callbacks to reduce training time by 50%+
- **End-to-end Pipeline**: Automated data preprocessing from raw images to normalized tensors

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model building and training
- **Scikit-Learn**: Machine learning utilities and metrics
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization and result plotting
- **Google Colab**: Development and training environment
- **NumPy**: Numerical computing operations

## 🏗️ Architecture

The model uses transfer learning with MobileNetV2 as the base architecture:

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: Additional dense layers for dog breed classification
- **Input Shape**: 224x224x3 RGB images
- **Output**: 120 classes (dog breeds)
- **Optimization**: Adam optimizer with custom learning rate scheduling

## 📊 Model Performance

- **Accuracy**: 68% on test dataset
- **Training Data**: 10,000+ dog images across 120 breeds
- **Training Time**: Optimized using TensorBoard callbacks (50% reduction)
- **Validation**: Comprehensive evaluation with confusion matrices and classification reports

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nryee2005/Dog_Vision.git
cd Dog_Vision
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook or run in Google Colab for best performance.

### Usage

1. **Data Preparation**: The notebook includes automated preprocessing pipeline
2. **Model Training**: Run the training cells to fine-tune the MobileNetV2 model
3. **Evaluation**: Analyze model performance using built-in metrics and visualizations
4. **Prediction**: Use the trained model to classify new dog images

## 📁 Project Structure

```
Dog_Vision/
├──Dog Vision/
   ├── train/                                          # Train data set
   ├── test/                                           # Test data set
   ├── models/                                         # Saved models 
   ├── logs/                                           # TensorBoard logs
   ├── custom images/                                  # Custom images used
   └── full_model_predictions_1_mobilenetV2.csv        # Predictions on test dataset using full model
├── dog_vision.ipynb                                # Jupyter notebook
├── requirements.txt                                # Project dependencies
└── README.md                                       # Project documentation
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- Deep learning and neural networks
- Transfer learning techniques
- Computer vision applications
- Data preprocessing and augmentation
- Model optimization and performance tuning
- TensorFlow/Keras framework usage

## 👨‍💻 Author

**Nathan Yee**
- Email: ryee.nathan@gmail.com
- LinkedIn: [nathan-r-yee](https://linkedin.com/in/nathan-r-yee)
- GitHub: [nryee2005](https://github.com/nryee2005)

---
