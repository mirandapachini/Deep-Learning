# Deep-Learning (Archive)

A collection of deep learning projects focused on medical imaging and neural networks, particularly chest X-ray analysis using the NIH ChestXray14 dataset.

## 📋 Project Overview

This repository contains multiple deep learning projects:

### 1. **Chest X-ray AI Portfolio Project** (`chest-xray-ai-portfolio/`)
A polished employer-facing deep learning application demonstrating chest X-ray abnormality detection and multi-label classification using the NIH ChestXray14 dataset.

This project was developed and iterated in Databricks, using Databricks Apps patterns and Unity Catalog Volume storage for the trained model artifacts. The app has also been updated to resolve the model and configuration files from the local workspace first, so it can run locally while still supporting the Databricks deployment path.

**Key Features:**
- Multi-class pathology detection (14 different chest conditions)
- Pre-trained Keras model for chest X-ray analysis
- GradCAM visualization for model interpretability
- Support for batch predictions and single image analysis
- Comprehensive logging and performance tracking
- Built and deployed using Databricks Apps

**Tech Stack:**
- TensorFlow/Keras for deep learning
- Streamlit for web interface
- Pandas for data handling
- Matplotlib & Seaborn for visualization
- SciPy for image processing
- Databricks for development and deployment

### 2. **Notebook & Documentation Assets** (`notebooks/` and `docs/`)
Supporting notebooks and project documentation for the training workflow, project write-up, and related analysis.

### 3. **MNIST Neural Network with Hidden Layers**
HTML documentation and implementation details for a neural network with hidden layers trained on the MNIST dataset.

### 4. **NIH ChestXray14 Full-Scale Training**
Documentation and training procedures for the full-scale NIH ChestXray14 model.

### 5. **NIH ChestXray14 Project Documentation**
Comprehensive documentation for the ChestXray14 project.

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip or conda package manager
- TensorFlow/Keras
- Streamlit

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mirandapachini/Deep-Learning.git
   cd Deep-Learning
   ```

2. **Install dependencies for the Chest X-ray AI portfolio app:**
   ```bash
   cd chest-xray-ai-portfolio
   pip install -r requirements.txt
   ```

3. **Required files:**
   - `chest-xray-ai-portfolio/data/cxr14_inference_model.keras` - Pre-trained model weights
   - `chest-xray-ai-portfolio/data/cxr14_classes.json` - Class labels for 14 chest pathologies
   - `chest-xray-ai-portfolio/data/cxr14_last_conv_layer.txt` - Configuration for GradCAM visualization

## 🏃 Running the Applications

### Chest X-ray AI Portfolio Project (Streamlit)

**Local Development:**
```bash
cd chest-xray-ai-portfolio
streamlit run app.py
```

The application will be available at `http://localhost:8501`

**Features:**
- Upload chest X-ray images (JPG, PNG)
- Get predictions for 14 different pathologies
- View confidence scores
- GradCAM visualizations to understand model predictions
- Prediction history and logging

### Notebook and Documentation Files
The notebooks and project documents are now organized under the `notebooks/` and `docs/` folders for easier navigation.

## 📊 Model Information

### NIH ChestXray14 Model
- **Input size:** 224×224 pixels
- **Output:** Multi-label classification for 14 chest conditions
- **Training data:** NIH ChestXray14 dataset
- **Framework:** TensorFlow/Keras

### Supported Pathologies
The model can detect the following conditions:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

## 📁 Project Structure

```
Deep-Learning/
├── README.md
├── chest-xray-ai-portfolio/
│   ├── app.py
│   ├── app.yaml
│   ├── requirements.txt
│   ├── data/
│   │   ├── cxr14_classes.json
│   │   ├── cxr14_inference_model.keras
│   │   ├── cxr14_last_conv_layer.txt
│   │   ├── test_catalog.csv
│   │   └── test_images/
│   ├── docs/
│   │   ├── GRADCAM_FIX_README.md
│   │   └── THRESHOLD_TUNING_INTEGRATION.md
│   └── .streamlit/
├── notebooks/
│   ├── MNIST Neural Network with Hidden Layers.ipynb
│   ├── MNIST Neural Network with Hidden Layers.html
│   ├── NIH ChestXray14 Full-Scale Training (1).ipynb
│   ├── NIH ChestXray14 Full-Scale Training.html
│   ├── NIH ChestXray14 Project Documentation (1).ipynb
│   └── NIH ChestXray14 Project Documentation.html
└── docs/
    └── DEPLOYMENT_GUIDE_Random_Gallery.md
```

## 📝 Documentation

- **notebooks/NIH ChestXray14 Project Documentation.html** - Complete project documentation
- **notebooks/NIH ChestXray14 Full-Scale Training.html** - Training procedures and methodology
- **notebooks/MNIST Neural Network with Hidden Layers.html** - Neural network architecture details
- **chest-xray-ai-portfolio/docs/GRADCAM_FIX_README.md** - GradCAM implementation notes
- **chest-xray-ai-portfolio/docs/THRESHOLD_TUNING_INTEGRATION.md** - Threshold optimization documentation
- **docs/DEPLOYMENT_GUIDE_Random_Gallery.md** - Deployment guidance

## 🔧 Configuration

### Model Paths
Models can be configured to load from different sources:
- Local file system
- Databricks Unity Catalog Volumes
- Cloud storage (AWS S3, Azure Blob Storage, etc.)

### Logging
The application maintains several logs:
- `cxr14_predictions_log.csv` - Prediction history
- `cxr14_error_log.csv` - Error tracking
- `cxr14_usage_log.csv` - Usage statistics
- `cxr14_performance_log.csv` - Performance metrics
- `cxr14_feedback_log.csv` - User feedback

## 🔬 Technical Details

### Image Processing
- Input images are preprocessed to 224×224 pixels
- Normalization applied for model input
- Support for JPG and PNG formats

### Model Output
- Confidence scores (0-1) for each pathology class
- GradCAM heatmaps for model interpretability
- Prediction uncertainty estimates

### Performance Monitoring
- Inference time tracking
- Model performance metrics
- Usage statistics and analytics

## 🐛 Troubleshooting

### Model Not Loading
- Ensure model file path is correct
- Check file permissions
- Verify model file integrity

### Streamlit Connection Issues
- Verify Streamlit is installed: `pip install streamlit`
- Check port 8501 is available
- Restart Streamlit if making configuration changes

### Image Upload Errors
- Supported formats: JPG, PNG
- Recommended size: 224×224 pixels
- Maximum file size: Check Streamlit configuration

## 📚 Resources

- [NIH ChestXray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [GradCAM Paper](https://arxiv.org/abs/1610.02055)

## 👤 Author

Miranda Pachini

## 📄 License

This project is part of a deep learning research initiative.

## 🤝 Contributing

Only collaborators are allowed to make changes to this repository.

For approved collaborators, please follow the standard Git workflow:
1. Create a feature branch
2. Commit your changes
3. Push to the branch
4. Create a pull request

## 📞 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
