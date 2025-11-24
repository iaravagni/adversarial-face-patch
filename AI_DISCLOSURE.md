# AI Systems Disclosure

## AI Models Used in This Research

# AI Systems Disclosure

## AI Models Used in This Research

# AI Systems Disclosure

## AI Models Used in This Research

### 1. Face Recognition Model (Attack Target)
**Type**: FaceNet (Pre-trained CNN from `facenet_pytorch`)
**Purpose**: Target system that adversarial patches are designed to fool
**Training Data**: Pre-trained on large face datasets; tested on LFW + custom team photos (with consent)
**Limitations**: Pre-trained model, vulnerable to adversarial attacks by design

---

### 2. Face Detection (MTCNN)
**Type**: Multi-task CNN from `facenet_pytorch`
**Purpose**: Detect faces before applying patches
**Use**: Preprocessing only

---

### 3. Adversarial Patch Generation (Attack)
**Type**: Gradient-based optimization (NOT a trained model)
**Method**: 
- Initialize random circular patch
- Optimize patch pixels to maximize FaceNet misclassification
- Uses backpropagation and gradient descent
**Data**: LFW + custom team photos
**Note**: Patches are GENERATED, not trained

---

### 4. Patch Detector (Defense - Test)
**Type**: Custom CNN binary classifier (`PatchDetector`)
**Purpose**: Detect if image contains adversarial patch
**Training**: 
- PyTorch with Adam optimizer
- 50% clean images, 50% patched images from LFW
- Binary classification: Clean vs. Patch
**Performance**: 85-95% detection accuracy
**Limitations**: Only detects circular patches similar to training data

---

### 5. Adversarial Training (Defense - Train)
**Type**: Robust face recognition CNN
**Purpose**: Face recognition that works despite patches
**Training**: Standard face recognition but 50% training images include patches
**Performance**: -5% on clean images, +30-40% on patched images
**Trade-off**: Slightly less accurate on clean images, much more robust overall


## Transparency Statements

### Data Collection & Use
- **Consent**: All personal images used with explicit consent or from open source datasets
- **Privacy**: No unauthorized collection or use of facial data
- **Storage**: Data stored securely, used only for this research

### Model Limitations
- Models are **not production-ready**
- **Not suitable** for real-world security applications
- Should **not be used** for actual authentication

### Potential Harms & Risks

**Risks of Attack Models**:
- Could be misused to evade legitimate security systems
- Privacy implications if used without authorization

**Mitigation Strategies**:
- Code released with educational license only
- Focus on defense mechanisms
- Clear ethical guidelines
- Responsible disclosure practices

## AI-Generated Content Disclosure

### Use of AI Assistants
**Tools Used**: Claude AI, ChatGPT

**How AI Was Used**:
- Code suggestions and debugging
- Documentation writing
- Conceptual explanations
- Designing Algorithms

**What AI Did NOT Do**:
- All experiments conducted by team members
- All conclusions are our own

## Accountability

**Research Team**:
- Shefali Ahuja
- Iara Ravagni
- Lalit Lakamsani

**Course**: CYBERSEC 590 : AI for Offensive and Defensive Cybersecurity
**Institution**: Duke University

## Compliance

This research complies with:
-  Responsible AI principles

**Last Updated**: November 23, 2025
