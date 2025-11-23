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
- Performance may vary with lighting, angles, and image quality

**Limitations**:
- Not tested on production-grade systems
- Simplified models for educational purposes
- May not generalize to all real-world scenarios

---

### 2. Patch Detector (Defense Model)
**Type**: CNN-based binary classifier
**Purpose**: Detect presence of adversarial patches in images
**Training Data**: Synthetic adversarial patches + face images

**Known Limitations**:
- May not detect all types of adversarial attacks
- Trained on specific patch patterns
- False positives/negatives possible
- Requires retraining for new attack methods

---

## Transparency Statements

### Data Collection & Use
-
- **Consent**: All personal images used with explicit consent
- **Privacy**: No unauthorized collection or use of facial data
- **Storage**: Data stored securely, used only for this research
- **Retention**: Data will be deleted after project completion

### Model Limitations
- Models are **not production-ready**
- **Not suitable** for real-world security applications
- Should **not be used** for actual authentication or surveillance
- Performance metrics are for **research purposes only**

### Potential Harms & Risks

#### Risks of Attack Models:
- Could be misused to evade legitimate security systems
- Privacy implications if used without authorization
- Potential for identity fraud if misapplied

#### Mitigation Strategies:
- Code released with educational license only
- Focus on defense mechanisms alongside attacks
- Clear ethical guidelines and disclaimers
- Responsible disclosure practices

#### Risks of Defense Models:
- False sense of security if deployed without proper testing
- May not catch sophisticated attacks
- Could be circumvented by adaptive adversaries

---

## Fairness & Bias Considerations

### Demographic Bias
- Face recognition systems may perform differently across:
  - Different skin tones
  - Ages
  - Genders
  - Facial features

### Our Approach:
- Acknowledge these biases exist
- Test on diverse dataset when possible
- Document performance variations
- Do not claim universal effectiveness

### What We DON'T Do:
- We do not claim our models work equally for all demographics
- We do not test on vulnerable populations
- We do not make claims about real-world deployments

---

## AI-Generated Content Disclosure

### Use of AI Assistants in This Project
**Tools Used**:  Claude, ChatGPT, GitHub Copilot

**How AI Was Used**:
- Code suggestions and debugging assistance
- Documentation and report writing help
- Conceptual explanations of techniques
- Literature review assistance

**What AI Did NOT Do**:
- All experiments conducted by team members
- All analysis and conclusions are our own
- Final code reviewed and understood by team

**Verification**:
- All AI-generated code was reviewed and tested
- We understand all code we submit
- We take full responsibility for our work

---

### Sustainability Considerations:
- Used pre-trained models where possible
- Efficient architectures to minimize compute
- Limited dataset size to reduce energy use

---

**Course**: CYBERSEC 590: AI For Offensive and Defensive Cybersecurity
**Instructor**: Brinnae Bent
**Institution**: Duke University

### Reporting Issues
If you identify:
- Ethical concerns with this research
- Potential misuse of techniques
- Errors or biases in our models
- Safety issues

Please contact: shefali.ahuja@duke.edu; iara.ravagni@duke.edu; lalit.lakamsani@duke.edu


---

## Version History

**v1.0** - November 2025
- Initial release
- Strategy 1 (Patch Detector) and Strategy 3 (Adversarial Training)
- Tested on LFW + custom team dataset

---

**Last Updated**: November 23, 2025

