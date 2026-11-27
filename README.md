# TriGuard: A Multimodal Lip–Voice Synchronization Based Liveness Authentication System

<p align="center">
  <strong>Research-Based AI/ML Capstone Project</strong>
</p><p align="center">
  <img src="https://img.shields.io/badge/Domain-Artificial%20Intelligence-blue" />
  <img src="https://img.shields.io/badge/Research-Multimodal%20Biometrics-purple" />
  <img src="https://img.shields.io/badge/Language-Python-yellow" />
  <img src="https://img.shields.io/badge/Interface-PyQt5-green" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-orange" />
</p>---

Abstract

Biometric authentication systems based on facial recognition and voice recognition are increasingly used for identity verification. However, conventional unimodal authentication methods may be vulnerable to presentation attacks, including recorded video replay, voice replay, and synthetic media.

This research presents TriGuard, a multimodal biometric authentication framework that combines facial recognition, speaker verification, and visual speech-based liveness analysis.

The proposed framework captures audio and video simultaneously in response to a randomly selected speech challenge. The captured data are processed through three independent authentication modules: a face recognition module, a speaker verification module, and a lip-based visual speech module.

The outputs of these modules are integrated using a weighted parallel fusion strategy. The final authentication decision considers the combined confidence score, lip-based liveness verification, and agreement between facial and voice identities.

The objective is to investigate whether combining multiple biometric characteristics and challenge-based verification can improve authentication reliability compared with relying on a single biometric modality.

Keywords: Multimodal Biometrics, Liveness Detection, Speaker Verification, Face Recognition, Visual Speech Recognition, Lip–Voice Synchronization, Presentation Attack Detection.

1. Introduction

Biometric authentication is an important technology for verifying the identity of individuals using unique physical or behavioural characteristics.

Common biometric techniques include facial recognition, fingerprint recognition, and voice recognition. Although these techniques provide convenient authentication, individual biometric systems may be vulnerable to spoofing and presentation attacks.

For example, an attacker may attempt to authenticate using a recorded face video, replayed voice recording, or manipulated audio-visual content.

To address these challenges, this research proposes TriGuard, a multimodal authentication framework that combines three biometric components:

1. Facial identity verification.
2. Voice-based speaker verification.
3. Lip-based visual speech and challenge verification.

By analysing multiple biometric signals within the same authentication session, TriGuard aims to provide an additional layer of security against unauthorized access.

2. Problem Statement

Traditional biometric authentication systems often depend on a single source of biometric information.

Such systems may face security limitations when attackers use recorded or artificially generated biometric samples.

The research problem is to develop a multimodal authentication framework capable of analysing facial identity, speaker identity, and visual speech responses together to improve the reliability of liveness-based authentication.

3. Research Objectives

- To develop a multimodal biometric authentication framework.
- To implement facial recognition using deep-learning-based face embeddings.
- To implement speaker verification using voice embeddings.
- To investigate visual speech recognition for challenge-based liveness verification.
- To capture audio and video simultaneously during authentication.
- To implement weighted score-level fusion.
- To evaluate the authentication framework using appropriate performance metrics.

4. Proposed Methodology

TriGuard follows a multimodal authentication methodology consisting of the following stages.

4.1 Random Challenge Generation

At the beginning of an authentication session, the system selects a sentence randomly from a predefined collection of challenge sentences.

The user is required to speak the displayed sentence during the recording session.

This challenge-response mechanism is intended to reduce dependence on predictable, previously recorded responses.

4.2 Audio-Visual Data Acquisition

The system captures audio and video simultaneously using a microphone and camera.

The captured information includes:

- Facial video frames.
- Audio speech recording.
- Selected challenge sentence.

The same captured audio recording is provided to the voice and lip-related processing modules.

4.3 Facial Recognition Module

The facial recognition module verifies the identity of the person appearing in the captured video.

The implementation uses:

- MTCNN for face detection.
- ArcFace for facial feature extraction.
- Face embeddings for identity comparison.

The extracted facial embedding is compared with registered identity embeddings using a configured distance threshold.

4.4 Voice Verification Module

The voice module verifies the speaker's identity using speech characteristics.

The implementation uses an ECAPA-based speaker embedding model associated with the SpeechBrain speaker verification framework.

The voice processing pipeline includes:

1. Audio input acquisition.
2. Audio preprocessing and sample-rate handling.
3. Speaker embedding extraction.
4. Comparison with registered voice embeddings.
5. Cosine similarity-based identity verification.

The similarity score is compared with the configured acceptance threshold.

4.5 Lip-Based Visual Speech and Liveness Module

The lip module analyses the captured visual speech response using a visual speech recognition model.

The module uses a pretrained visual speech model checkpoint and evaluates the relationship between the observed visual speech and the expected challenge response.

The implementation includes challenge-text matching and configurable liveness thresholds.

This component is used as a liveness gate within the overall authentication decision.

4.6 Multimodal Score Fusion

The three authentication modules operate in parallel.

Each module produces an authentication result and a corresponding confidence score.

The configured weights are:

Module| Weight
Face recognition| 0.40
Voice verification| 0.35
Lip liveness| 0.25

The weighted authentication score is calculated as:

[
S_{final}=0.40S_f+0.35S_v+0.25S_l
]

Where:

- (S_f) = Face module score.
- (S_v) = Voice module score.
- (S_l) = Lip module score.

The final decision uses a configured authentication threshold of 0.60.

Additionally, authentication requires the lip module to pass its liveness condition and, when both face and voice modules are active, their identified users must agree.

The fusion mechanism therefore combines score-level evidence with additional decision-level security conditions.

5. System Architecture

flowchart TD
    A[User Authentication] --> B[Random Challenge Selection]
    B --> C[Countdown and Audio-Visual Capture]

    C --> D[Captured Video]
    C --> E[Captured Audio]

    D --> F[Face Detection and Recognition]
    D --> G[Visual Speech / Lip Analysis]

    E --> H[Speaker Verification]

    F --> I[Face Identity Score]
    G --> J[Lip Liveness Score]
    H --> K[Voice Identity Score]

    I --> L[Weighted Score Fusion]
    J --> L
    K --> L

    L --> M{Authentication Conditions}
    M -->|Passed| N[Access Granted]
    M -->|Failed| O[Access Denied]

6. System Workflow

1. The application initializes the authentication modules.
2. The user starts an authentication session.
3. A random challenge sentence is selected.
4. A countdown is displayed before recording.
5. Audio and video are recorded simultaneously.
6. The face module performs facial identity verification.
7. The voice module performs speaker verification.
8. The lip module performs visual speech and liveness analysis.
9. The orchestrator collects the module results.
10. Weighted score fusion is performed.
11. Additional identity and liveness conditions are checked.
12. The application displays the final authentication result.

7. Technologies and Frameworks

Technology| Application
Python| Core implementation
PyQt5| Desktop application interface
OpenCV| Video acquisition and processing
MTCNN| Face detection
ArcFace| Facial feature extraction
SpeechBrain| Speaker verification framework
PyTorch| Deep-learning model execution
TensorFlow| Face-module dependencies
MediaPipe| Visual processing dependencies
NumPy| Numerical operations
SciPy| Signal processing
FFmpeg| Audio/video processing support

8. Dataset and Model Resources

The research involves audio-visual biometric data and pretrained model resources.

Potential dataset resources for model development and evaluation include:

- GRID Corpus: Audio-visual speech recordings for studying speech and lip movement.
- LRS3: Large-scale audio-visual speech data.
- ASVspoof: Spoofed and genuine speech data for audio anti-spoofing research.
- Self-Collected Samples: Live recordings for application-level testing.

The actual dataset split, sample counts, training procedures, and model-specific dataset usage should be documented according to the experiments performed.

Model Resources

The implementation references pretrained model resources, including:

- ArcFace facial recognition model.
- ECAPA-based speaker embedding model.
- Visual speech recognition checkpoint.

Large model checkpoints and sensitive datasets are not distributed as part of the source-code repository. Their installation and access requirements should be documented separately.

9. Experimental Evaluation

The proposed system should be evaluated through individual module testing and multimodal authentication testing.

Evaluation Metrics

Metric| Purpose
Accuracy| Overall correct classification rate
Precision| Correct positive predictions
Recall| Ability to identify genuine positive samples
F1-score| Balance between precision and recall
FAR| False Acceptance Rate
FRR| False Rejection Rate
EER| Equal Error Rate

Experimental Results

The final experimental results should be reported separately for:

1. Facial recognition.
2. Speaker verification.
3. Lip-based liveness verification.
4. Combined multimodal authentication.

Results table – To be completed with final verified experimental measurements.

Experiment| Accuracy| Sample Count
Face module| To be measured| To be added
Voice module| To be measured| To be added
Lip module| To be measured| To be added
Multimodal fusion| To be measured| To be added

The evaluation should include genuine-user and attack samples, with clearly defined testing conditions and data separation.

10. Limitations

- Authentication performance depends on camera and microphone quality.
- Lighting, background noise, and recording conditions may affect predictions.
- The performance of visual speech analysis depends on the quality and compatibility of model resources.
- Challenge-text matching alone cannot establish resistance to every deepfake or presentation attack.
- The configured fusion weights and decision thresholds require systematic evaluation.
- Performance on unseen users and unseen spoofing methods requires further investigation.

11. Future Research Scope

- Improve robustness against AI-generated audio and video.
- Investigate advanced audio-visual synchronization techniques.
- Explore phoneme-viseme alignment for more detailed synchronization analysis.
- Evaluate transformer-based multimodal architectures.
- Investigate advanced presentation attack detection methods.
- Expand evaluation using diverse datasets and unseen attack scenarios.
- Optimize model inference for real-time authentication.
- Explore secure mobile and web-based deployment.

12. Repository Structure

TriGuard_Syatem/
│
├── core/
│   ├── capture_session.py
│   ├── orchestrator.py
│   ├── config_loader.py
│   └── logger.py
│
├── modules/
│   ├── face_module.py
│   ├── voice_module.py
│   └── lip_module.py
│
├── data/
│   ├── lip_weights/
│   ├── pretrained_models/
│   ├── face_embeddings/
│   └── voice_dataset/
│
├── datamodule/
├── preparation/
├── testing/
├── tools/
├── webapp/
│
├── config.json
├── requirements.txt
├── main.py
├── gui_main.py
├── lip_track.py
└── README.md

This is a logical overview of the main project components; some data and model directories may be created locally or supplied separately.

13. Installation and Execution

Prerequisites

- Python environment compatible with the listed dependencies.
- Camera and microphone.
- Required pretrained model checkpoints.
- Appropriate CPU/GPU runtime.
- FFmpeg installed and accessible.

Step 1: Clone the Repository

git clone https://github.com/Pradip-4/TriGuard_Syatem.git

Step 2: Open the Project Directory

cd TriGuard_Syatem

Step 3: Create a Virtual Environment

python -m venv venv

Step 4: Activate the Environment

Windows:

venv\Scripts\activate

Step 5: Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Step 6: Configure Model Resources

Verify that the required model checkpoints, registered face embeddings, voice embeddings, and configuration paths are available.

Review "config.json" before execution, particularly the model paths, hardware device, and authentication thresholds.

Step 7: Run the Desktop Application

python gui_main.py

Alternatively, run the command-line authentication entry point:

python main.py

14. Research Contributions

The main research contributions of this project are:

- A multimodal biometric authentication architecture combining facial, voice, and visual speech information.
- A challenge-response mechanism for unpredictable speech verification.
- Parallel execution of independent authentication modules.
- Weighted score-level fusion with additional liveness and identity-consistency conditions.
- An experimental framework for investigating multimodal biometric authentication.

These contributions describe the implemented framework and research direction; their effectiveness must be established through controlled experimental evaluation.

15. Conclusion

TriGuard presents a multimodal biometric authentication framework that integrates facial recognition, speaker verification, and visual speech-based liveness analysis.

By combining different biometric characteristics and a randomized challenge-response mechanism, the system aims to address certain limitations of conventional single-modality authentication.

The weighted fusion strategy integrates the outputs of individual modules, while additional liveness and identity-consistency conditions contribute to the final authentication decision.

Further experimental evaluation using diverse datasets and realistic presentation attack scenarios is necessary to establish the security and generalization capabilities of the proposed framework.

16. Project Information

Attribute| Details
Project Title| TriGuard
Project Domain| Artificial Intelligence and Machine Learning
Research Area| Multimodal Biometric Authentication
Project Type| Final-Year Engineering Capstone
Primary Language| Python
Repository| TriGuard_Syatem

17. Developer

Pradip Patil

GitHub: https://github.com/Pradip-4

Project Repository: https://github.com/Pradip-4/TriGuard_Syatem

---

References

Add the exact research papers and model publications used in the final implementation and literature review.

Suggested reference areas:

1. ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
2. ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.
3. Visual Speech Recognition and Lip Reading.
4. GRID Corpus and LRS3 audio-visual speech datasets.
5. ASVspoof challenge research on automatic speaker verification spoofing and countermeasures.

Full bibliographic details, publication years, authors, and DOI links should be verified before academic submission.

---

Note: TriGuard is an academic research prototype. Its authentication outputs should not be interpreted as proof of resistance to all biometric spoofing or deepfake attacks.
