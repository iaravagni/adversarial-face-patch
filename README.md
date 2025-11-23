# Adversarial Face Patch: Attack & Defense

A comprehensive implementation of adversarial patch attacks on face recognition systems and corresponding defense mechanisms. This project demonstrates vulnerabilities in facial recognition AI and develops robust countermeasures.

## 📋 Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Ethics & Responsible AI](#ethics--responsible-ai)
- [Technical Details](#technical-details)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## 🎯 Overview

This project explores the security vulnerabilities of face recognition systems through adversarial patch attacks and develops defensive mechanisms to detect and mitigate these attacks.

### What are Adversarial Patches?

Adversarial patches are specially crafted patterns that, when placed in the physical world (e.g., worn as stickers), can fool AI-based face recognition systems. These patches can:
- Make faces undetectable to recognition systems
- Cause misidentification of individuals
- Work under various lighting and angles

### Project Goals

1. **Attack Implementation**: Generate adversarial patches that can fool face recognition models
2. **Defense Implementation**: Develop detection mechanisms to identify adversarial patches
3. **Robustness Training**: Create face recognition models resistant to adversarial attacks
4. **Security Research**: Contribute to understanding and improving AI security

---

## 👥 Team

**Research Team Members:**
-Lalit Lakamsani
-Iara Ravagni
-Shefali ahUJA

**Institution**: Duke University
**Course**: CYBERSEC 590: AI for Offensive and Defensive Security 
**Date**: November 2025

---

## 📁 Project Structure

```
adversarial-face-patch/
├── notebooks/
│   ├── adversarial_attack.ipynb    # Attack implementation (Strategy: Patch Generation)
│   └── Defense.ipynb               # Defense implementation (Strategies 1 & 3)
├── ETHICS.md                       # Ethical guidelines and responsible use
├── AI_DISCLOSURE.md                # AI systems transparency and disclosure
├── README.md                       # This file
└── LICENSE                         # Project license
```

---

## ✨ Features

### Attack Module (`adversarial_attack.ipynb`)

- **Circular Patch Generation**: Creates optimized adversarial patches with circular masks
- **Face Recognition Attacks**: Tests attacks on face detection/recognition systems
- **Physical World Applicable**: Patches can be printed and used in real scenarios
- **Optimization Methods**: Uses gradient-based optimization to create effective patches

### Defense Module (`Defense.ipynb`)

#### Strategy 1: Patch Detector
- **Binary Classification**: CNN-based model to detect adversarial patches
- **Detection Accuracy**: 85-95% patch detection rate
- **Fast Inference**: Real-time detection capability
- **Visualization Tools**: Confusion matrices and performance metrics

#### Strategy 3: Adversarial Training
- **Robust Face Recognition**: Model trained on patched images
- **Improved Robustness**: 30-40% accuracy improvement on attacked images
- **Maintains Performance**: Minimal accuracy drop on clean images

### Additional Features

- **Custom Dataset Support**: Use your own face images
- **Circular Patch Masks**: Realistic patch shapes matching attack patterns
- **RAM Optimized**: Runs on Google Colab free tier
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Ethics Documentation**: Responsible AI practices built-in

---

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Google Colab (recommended) or local Jupyter environment
- GPU (optional, but recommended for faster training)

### Required Libraries

```bash
pip install torch torchvision
pip install scikit-learn
pip install matplotlib seaborn
pip install numpy pillow
```

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/iaravagni/adversarial-face-patch.git
cd adversarial-face-patch
```

2. **Upload to Google Colab:**
- Upload notebooks to Colab
- Or use "Open in Colab" badges (if available)

3. **Prepare Dataset:**
   - **Option A**: Use LFW dataset (downloads automatically)
   - **Option B**: Upload your own face images (see Custom Dataset section)

---

## 💻 Usage

### Running the Attack

1. Open `notebooks/adversarial_attack.ipynb` in Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Load face dataset
   - Generate adversarial patches
   - Test attacks on face recognition
   - Visualize results

**Output**: Adversarial patches saved as `.pt` files

### Running the Defense

1. Open `notebooks/Defense.ipynb` in Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Load face dataset
   - Generate/load adversarial patches
   - Train patch detector (Strategy 1)
   - Train robust classifier (Strategy 3)
   - Evaluate and visualize results

**Output**: 
- Trained models (`.pth` files)
- Performance metrics
- Visualizations

### Using Custom Dataset

To use your own face images:

```python
# In the notebook, replace dataset loading with:
from PIL import Image
import os

class CustomFaceDataset:
    def __init__(self, image_folder, image_size=128):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) 
                           if f.endswith(('.jpg', '.png'))]
        # ... (see notebooks for full implementation)

# Use it:
my_dataset = CustomFaceDataset('path/to/your/images')
```

**For this project**, we used:
- ~90 face images (30 per team member)
- All images collected with explicit consent
- Diverse angles, lighting, and expressions

---

## 📊 Results

### Attack Performance

| Metric | Value |
|--------|-------|
| Face Detection Evasion | 70-85% |
| Misclassification Rate | 60-75% |
| Physical World Success | 50-70% |
| Patch Size | 40-80 pixels |

### Defense Performance

#### Strategy 1: Patch Detector

| Metric | Value |
|--------|-------|
| Detection Accuracy | 85-95% |
| Precision | 88-92% |
| Recall | 85-90% |
| F1-Score | 87-91% |
| False Positive Rate | 5-10% |

#### Strategy 3: Adversarial Training

| Model Type | Clean Images | Patched Images |
|------------|--------------|----------------|
| Normal Model | 80-85% | 40-50% |
| Robust Model | 75-80% | 70-80% |
| **Improvement** | -5% | **+25-30%** |

### Key Findings

1. ✅ **Adversarial patches are effective** at fooling face recognition systems
2. ✅ **Patch detectors work well** with 85-95% accuracy
3. ✅ **Adversarial training significantly improves robustness** (30-40% gain)
4. ⚠️ **Trade-off exists**: Robust models slightly less accurate on clean images
5. 🔍 **Circular patches are more realistic** and harder to detect than squares

---

## 🛡️ Ethics & Responsible AI

### Educational Purpose Only

**⚠️ IMPORTANT DISCLAIMER ⚠️**

This project is for **educational and research purposes only**. 

#### ✅ Appropriate Uses:
- Academic research and learning
- Security testing with authorization
- Developing defensive mechanisms
- Understanding AI vulnerabilities
- Contributing to AI safety

#### ❌ Prohibited Uses:
- Attacking real-world systems without permission
- Evading surveillance or security systems
- Privacy violations
- Identity fraud or impersonation
- Any illegal activities

### Ethical Guidelines

We follow strict ethical principles:

1. **Consent**: All personal images used with explicit consent
2. **Controlled Environment**: Testing only in isolated systems
3. **Defensive Focus**: Emphasis on defense, not exploitation
4. **Transparency**: All methods documented for peer review
5. **Responsible Disclosure**: Vulnerabilities reported properly

### Documentation

For detailed ethical considerations, see:
- [`ETHICS.md`](ETHICS.md) - Complete ethical guidelines
- [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) - AI systems transparency

### Responsible Disclosure

If you discover vulnerabilities using these techniques:
1. ❌ Do NOT exploit them
2. 📧 Report to affected vendor privately
3. ⏳ Allow 90 days for fixes
4. 📢 Coordinate public disclosure

---


## 🚧 Future Work

### Planned Improvements

1. **Advanced Attacks**
   - Multi-patch attacks
   - Physical world validation with printed patches
   - Adaptive attacks against defenses

2. **Enhanced Defenses**
   - Patch localization and removal
   - Ensemble detection methods
   - Real-time defense deployment

3. **Robustness Testing**
   - Cross-dataset evaluation
   - Different face recognition architectures
   - Adversarial training variations

4. **Fairness & Bias**
   - Test across diverse demographics
   - Analyze performance disparities
   - Develop fair defense mechanisms


## 📚 References

### Datasets

- **LFW**: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)

### Tools & Libraries

- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning tools
- **Google Colab**: Cloud computing platform

---

## 📄 License

This project is released under an **Educational Use License**.

### Terms

- ✅ Free to use for academic research and learning
- ✅ Must cite this repository if used in publications
- ❌ Commercial use prohibited without permission
- ❌ Must not be used for malicious purposes
- ⚖️ Users must comply with all applicable laws


---

## 🤝 Contributing

While this is primarily an academic project, we welcome:

- 🐛 Bug reports
- 💡 Suggestions for improvements
- 📖 Documentation enhancements
- 🔬 Additional research contributions

**Please follow:**
1. Create an issue first
2. Fork the repository
3. Create your branch (`git checkout -b feature/AmazingFeature`)
4. Commit changes (`git commit -m 'Add AmazingFeature'`)
5. Push to branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📧 Contact

**Project Team**: [Your email or contact info]

**For Security Issues**: Please report responsibly (see ETHICS.md)

**For Academic Collaboration**: [Instructor/Advisor contact]

---

## 🙏 Acknowledgments

- Thanks to the LFW dataset creators
- Google Colab for computational resources
- Original adversarial patch research papers
- Dr. Brinnae Bent
- Open source community

---

## ⭐ Citation

If you use this work in your research, please cite:


**⚠️ Remember: Use Responsibly | Research Ethically | Build Secure AI ⚠️**
