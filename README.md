# 🐄 Cattle Breed AI Classification Model# Model_cattleBreed_RSNET18

A production-ready deep learning model for classifying cattle breeds using ResNet18 architecture.

## 🎯 Model Performance
- **Architecture**: ResNet18
- **Accuracy**: 93.24% validation accuracy  
- **Classes**: 124 cattle breeds
- **Input**: 224x224 RGB images
- **Output**: Top 3 breed predictions with confidence scores

## 📁 Project Structure

```
cattle-breed-ai/
├── 📄 README.md                    # This documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📁 src/                         # Source code
│   └── predict_cattle.py           # Main prediction script
├── 📁 models/                      # Model files (see download section)
├── 📁 scripts/                     # Training utilities
│   ├── stable_gpu_train.py         # Model training script
│   └── enhance_cattle_dataset.py   # Dataset enhancement
├── 📁 config/                      # Configuration files
│   └── cattle_dataset.yaml         # Breed class mappings
└── 📁 tests/                       # Test files (future)
```

## 📦 Model Download

**⚠️ Important**: The trained model file is too large for GitHub (128MB).

**Get the model file:**
- **Contact**: [@mohvijayjain](https://github.com/mohvijayjain) for the trained model
- **File**: `stable_cattle_model.pth` (128MB)
- **Place in**: `models/` directory
- **Alternative**: Train your own using `scripts/stable_gpu_train.py`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Model File
Download `stable_cattle_model.pth` and place in `models/` folder

### 3. Run Prediction
```bash
cd src
python predict_cattle.py path/to/your/cattle_image.jpg
```

### Expected Output
```
🐄 Loading Cattle Breed Predictor...
🖥️ Using device: cuda
✅ Model loaded successfully!
🧠 Trained to recognize 124 cattle breeds
🏆 Best validation accuracy: 93.24%

🖼️ Analyzing image: your_cattle_image.jpg
🏆 Top 3 Cattle Breed Predictions:
🥇 1. Halikar - Confidence: 100.0%
🥈 2. Hallikar - Confidence: 0.0%  
🥉 3. Kangayam - Confidence: 0.0%
```

## 🛠️ Integration Usage

### Python Integration
```python
from src.predict_cattle import CattlePredictor

# Initialize predictor
predictor = CattlePredictor('models/stable_cattle_model.pth')

# Get predictions
results = predictor.predict_top3('path/to/image.jpg')

# Results format:
# [
#   {'breed': 'Halikar', 'confidence': 1.0, 'percentage': '100.0%'},
#   {'breed': 'Hallikar', 'confidence': 0.0, 'percentage': '0.0%'},
#   {'breed': 'Kangayam', 'confidence': 0.0, 'percentage': '0.0%'}
# ]
```

### Web API Example
```python
from flask import Flask, request, jsonify
from src.predict_cattle import CattlePredictor

app = Flask(__name__)
predictor = CattlePredictor('models/stable_cattle_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp.jpg')
    results = predictor.predict_top3('temp.jpg')
    return jsonify({'predictions': results})
```

## 📦 Dependencies

- torch>=1.9.0
- torchvision>=0.10.0  
- Pillow>=8.0.0
- PyYAML>=5.4.0

## 🔧 Training Your Own Model

```bash
cd scripts
python stable_gpu_train.py
```

**Requirements for training:**
- CUDA-capable GPU
- Cattle breed dataset (124+ classes)
- Images organized by breed folders

## 🎯 Integration Ready

This model is production-ready for:
- ✅ Web applications (Flask/Django/FastAPI)
- ✅ Mobile apps (via ONNX conversion)  
- ✅ Desktop applications
- ✅ REST APIs
- ✅ Cloud services (AWS/GCP/Azure)

## 📊 Technical Details

- **Framework**: PyTorch
- **Model Size**: ~85MB
- **Inference Speed**: ~50ms on GPU, ~200ms on CPU
- **Memory Usage**: ~2GB GPU memory during inference
- **Input Format**: RGB images, any size (auto-resized to 224x224)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohit Vijay Jain** - [@mohvijayjain](https://github.com/mohvijayjain)

## 🔗 Links

- **Repository**: [Model_cattleBreed_RSNET18](https://github.com/mohvijayjain/Model_cattleBreed_RSNET18)
- **Issues**: [Report bugs](https://github.com/mohvijayjain/Model_cattleBreed_RSNET18/issues)
- **Contact**: For model file access and support

---

⭐ **Star this repository if you found it helpful!**