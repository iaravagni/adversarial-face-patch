# AI Systems Disclosure

## AI Models Used in This Research

### 1. Face Recognition Models
**Type**: Convolutional Neural Networks (CNNs)
**Purpose**: Target systems for adversarial attacks and defense testing
**Training Data**: 
- LFW (Labeled Faces in the Wild) dataset
- Custom dataset of research team members (with consent)

**Potential Biases**:
- May perform differently across different demographics
- Training data may not represent all populations equally

**Limitations**:
- Not tested on production-grade systems
- Simplified models for educational purposes

### 2. Patch Detector (Defense Model)
**Type**: CNN-based binary classifier
**Purpose**: Detect presence of adversarial patches
**Training Data**: Synthetic adversarial patches + face images

**Known Limitations**:
- May not detect all types of adversarial attacks
- Trained on specific patch patterns
- Requires retraining for new attack methods

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
