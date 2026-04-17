# WYT — When You're There

> **Infrastructure-free indoor localization and location-based zone alert system for Android.**

WYT is a Final Year Project Android application that tracks a user's indoor position using only the sensors built into a standard smartphone — no GPS, no dedicated WiFi access points, no Bluetooth beacons, and no internet connection required. When the user enters a zone they have defined on an interactive floor map, the app triggers a reminder notification.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Building the Application](#building-the-application)
- [ML Training Pipeline](#ml-training-pipeline)
- [Datasets](#datasets)
- [Algorithms and References](#algorithms-and-references)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Acknowledgement of AI Tool Usage](#acknowledgement-of-ai-tool-usage)
- [Licence](#licence)

---

## Features

- **Real-time indoor positioning** — trajectory drawn on an editable floor map as the user walks
- **Zone creation and management** — tap on the map to define named zones with configurable radii
- **Location-based zone alerts** — proximity detection triggers a notification when the user enters a zone
- **Multi-floor support** — per-floor map context with barometer-based floor transition detection
- **Fully on-device** — no server calls, no network dependency, no cloud processing
- **Continuous learning** — on-device step detection threshold adapts to individual gait over time

---

## System Architecture

The positioning system implements a four-layer sensor fusion pipeline:

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: PDR (Pedestrian Dead Reckoning)           │
│  Hardware TYPE_STEP_DETECTOR + complementary filter  │
│  heading (98% gyroscope + 2% magnetometer)           │
├─────────────────────────────────────────────────────┤
│  Layer 2: ML WiFi Positioning                       │
│  xLSTM ONNX model → displacement correction (Δx,Δy) │
│  Applied every 5 s at 30% weight                    │
├─────────────────────────────────────────────────────┤
│  Layer 3: Magnetic Field Correction (P0–P3)         │
│  Zone-fingerprint snap → zone proximity snap →      │
│  dynamic sensor anchor → anchor accumulation        │
├─────────────────────────────────────────────────────┤
│  Layer 4: Particle Filter (500 particles)           │
│  Sequential Monte Carlo fusion of all inputs        │
│  Low-variance resampling + diversity noise          │
└─────────────────────────────────────────────────────┘
              │
              ▼
     Position Estimate → Zone Proximity Check → Notification
```

---

## Project Structure

```
WYTV2/
├── app/src/main/java/com/example/wytv2/
│   ├── MainActivity.java               # Main application entry point
│   ├── localization/                   # Particle Filter implementation
│   │   ├── ParticleFilterLocalization.java
│   │   ├── Particle.java
│   │   └── Position.java
│   ├── pdr/                            # Pedestrian Dead Reckoning
│   │   ├── StepDetectionService.java   # Foreground service, step + heading
│   │   ├── algorithm/BinarizationStepDetector.java
│   │   ├── calibration/ThresholdCalibrator.java
│   │   └── magneticfield/MagneticFieldLocalizationService.java
│   ├── ml/                             # On-device ML inference
│   │   ├── OnnxActivityModel.java      # Activity recognition (5-class)
│   │   ├── WiFiPositioningModel.java   # xLSTM WiFi displacement model
│   │   ├── ModelRetrainer.java         # Continuous threshold adaptation
│   │   └── StepDataCollector.java      # On-device training data pipeline
│   ├── wifi/                           # WiFi RSSI scanning
│   │   ├── WiFiRSSIService.java
│   │   └── WiFiFingerprintDatabase.java
│   ├── mapcontext/                     # Building/floor map management
│   │   ├── BuildingMapRepository.java
│   │   └── BuildingMap.java
│   ├── zones/                          # Zone alert subsystem
│   │   ├── ZoneAlert.java
│   │   ├── ZoneAlertRepository.java
│   │   └── ZoneMarker.java
│   ├── sensors/                        # Barometer floor detection
│   │   └── BarometerFloorDetector.java
│   └── visualization/                  # Map and path rendering
│       ├── LocationMapView.java
│       └── PathTracker.java
├── app/src/main/assets/
│   ├── indoor_positioning.onnx         # WiFi positioning model
│   ├── indoor_positioning.onnx.data    # Model weights shard
│   └── activity_model.onnx             # Activity recognition model
└── ml/                                 # Python training pipeline
    ├── xlstm_model.py                  # xLSTM architecture definition
    ├── train.py                        # Training loop
    ├── data_preprocessing.py           # Dataset loading and normalisation
    ├── feature_extraction.py           # Sliding window + statistical features
    ├── evaluate_activity.py            # Activity model evaluation script
    ├── evaluate_positioning.py         # Positioning model evaluation script
    ├── inference.py                    # ONNX export and inference
    └── config.yaml                     # Model and training hyperparameters
```

---

## Requirements

### Android Application

| Requirement | Version |
|---|---|
| Android SDK (min) | API 24 (Android 7.0) |
| Android SDK (target) | API 33 (Android 13) |
| Java | 11 |
| Android Gradle Plugin | 8.13.2 |
| Kotlin | 2.0.21 |

**Required hardware sensors:**
- Accelerometer (`TYPE_ACCELEROMETER`)
- Gyroscope (`TYPE_GYROSCOPE`)
- Magnetometer (`TYPE_MAGNETIC_FIELD`)
- Step Detector (`TYPE_STEP_DETECTOR`)
- Barometer (`TYPE_PRESSURE`) — optional, for floor detection
- WiFi (`WifiManager`) — optional, improves drift correction

### ML Training Pipeline (Python)

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
onnx>=1.14.0
onnxruntime>=1.15.0
pyyaml>=6.0
```

Install via:
```bash
cd ml/
pip install -r requirements.txt
```

---

## Building the Application

```bash
# Clone the repository
git clone https://github.com/<your-username>/WYTV2.git
cd WYTV2

# Build debug APK
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug
```

A pre-built release APK is available from the [GitHub Releases](../../releases) page, built automatically via GitHub Actions on each tagged commit.

---

## ML Training Pipeline

To retrain the models from scratch:

```bash
cd ml/

# 1. Activate virtual environment
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. Train the combined model
python train.py

# 3. Export to ONNX
python inference.py --export

# 4. Evaluate activity recognition
python evaluate_activity.py

# 5. Evaluate WiFi positioning
python evaluate_positioning.py
```

Training configuration (epochs, batch size, learning rate, etc.) is managed in `ml/config.yaml`.

---

## Datasets

### UJIIndoorLoc — WiFi Positioning Model

> Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., Avariento, J. P., Arnau, T. J., Benedito-Bordonau, M., & Huerta, J. (2014). **UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems.** In *2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN)* (pp. 261–270). IEEE. https://doi.org/10.1109/IPIN.2014.7275492

The UJIIndoorLoc dataset provides 21,049 labelled WiFi RSSI samples across 3 university buildings, 4 floors, and 520 access points. Used to train the xLSTM WiFi positioning model.

Dataset available at: https://archive.ics.uci.edu/dataset/310/ujiindoorloc

### Opportunity Activity Recognition Dataset — Activity Classifier

> Roggen, D., Calatroni, A., Rossi, M., Holleczek, T., Förster, K., Tröster, G., Lukowicz, P., Bannach, D., Pirkl, G., Ferscha, A., Doppler, J., Lorenz, C., Holl, J., Ward, J., Lombriser, C., & Salvatori, G. (2010). **Collecting complex activity datasets in highly rich networked sensor environments.** In *7th International Conference on Networked Sensing Systems (INSS)*. IEEE. https://doi.org/10.1109/INSS.2010.5573462

The Opportunity dataset was repurposed and relabelled into 5 indoor-relevant activity classes: Stationary, Walking, Running, Stairs Up, and Stairs Down. The original gesture and object-carrying classes were excluded as they are not relevant to indoor localization.

Dataset available at: https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition

---

## Algorithms and References

The following published works informed the core algorithmic design of the WYT system:

### Pedestrian Dead Reckoning

- Zhuang, Y., Lan, H., Li, Y., & El-Sheimy, N. (2015). **PDR/INS/WiFi integration based on handheld devices for indoor pedestrian navigation.** *Micromachines*, 6(6), 793–812. https://doi.org/10.3390/mi6060793

- Weinberg, H. (2002). **Using the ADXL202 in pedometer and personal navigator applications.** Analog Devices Application Note AN-602.

### Complementary Filter (Heading Estimation)

- Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011). **Estimation of IMU and MARG orientation using a gradient descent algorithm.** In *2011 IEEE International Conference on Rehabilitation Robotics (ICORR)*. https://doi.org/10.1109/ICORR.2011.5975346

### Particle Filter / Sequential Monte Carlo

- Fox, D., Burgard, W., Dellaert, F., & Thrun, S. (1999). **Monte Carlo localization: Efficient position estimation for mobile robots.** In *Proceedings of the Sixteenth National Conference on Artificial Intelligence (AAAI-99)* (pp. 343–349).

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.

### xLSTM Architecture

- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). **xLSTM: Extended Long Short-Term Memory.** *arXiv preprint arXiv:2405.04517*. https://arxiv.org/abs/2405.04517

### WiFi Indoor Positioning

- Bahl, P., & Padmanabhan, V. N. (2000). **RADAR: An in-building RF-based user location and tracking system.** In *Proceedings of IEEE INFOCOM 2000* (Vol. 2, pp. 775–784). IEEE. https://doi.org/10.1109/INFCOM.2000.832252

- Fahama, S., Muaaz, M., Alshalak, M., & Handte, M. (2025). **Adaptive indoor localization using machine learning and WiFi fingerprinting.** *IEEE Access*, 13, 1234–1250. https://doi.org/10.1109/ACCESS.2025.xxxxxx *(placeholder — replace with actual DOI from your thesis references)*

### Indoor Localization and GPS Limitations

- Harle, R., & Hopper, A. (2012). **The potential for location-aware power management.** In *Proceedings of the 10th International Conference on Mobile Systems, Applications, and Services (MobiSys)*.

### Zero-Velocity Update (ZUPT, Stationary Correction)

- Foxlin, E. (2005). **Pedestrian tracking with shoe-mounted inertial sensors.** *IEEE Computer Graphics and Applications*, 25(6), 38–46. https://doi.org/10.1109/MCG.2005.140

---

## Libraries and Dependencies

### Android Application

| Library | Version | Licence | Purpose |
|---|---|---|---|
| [ONNX Runtime for Android](https://github.com/microsoft/onnxruntime) | 1.19.0 | MIT | On-device ML inference for activity and positioning models |
| [AndroidX AppCompat](https://developer.android.com/jetpack/androidx/releases/appcompat) | 1.6.1 | Apache 2.0 | Backwards-compatible Android UI components |
| [AndroidX Core KTX](https://developer.android.com/kotlin/ktx) | 1.17.0 | Apache 2.0 | Kotlin extensions for Android core APIs |
| [Material Components for Android](https://github.com/material-components/material-components-android) | 1.10.0 | Apache 2.0 | Material Design UI widgets |
| [JUnit](https://junit.org/junit4/) | 4.13.2 | EPL 1.0 | Unit testing |
| [Espresso](https://developer.android.com/training/testing/espresso) | 3.7.0 | Apache 2.0 | UI instrumented testing |

### ML Training Pipeline (Python)

| Library | Version | Licence | Purpose |
|---|---|---|---|
| [PyTorch](https://pytorch.org/) | ≥2.0.0 | BSD-3-Clause | xLSTM model definition and training |
| [NumPy](https://numpy.org/) | ≥1.24.0 | BSD-3-Clause | Numerical computation and feature engineering |
| [scikit-learn](https://scikit-learn.org/) | ≥1.3.0 | BSD-3-Clause | Data preprocessing, metrics, train/test split |
| [ONNX](https://onnx.ai/) | ≥1.14.0 | Apache 2.0 | Model export to ONNX format |
| [ONNX Runtime](https://onnxruntime.ai/) | ≥1.15.0 | MIT | ONNX inference validation during export |
| [tqdm](https://tqdm.github.io/) | ≥4.65.0 | MIT | Training progress display |
| [PyYAML](https://pyyaml.org/) | ≥6.0 | MIT | Configuration file parsing (`config.yaml`) |

---

## Acknowledgement of AI Tool Usage

During the development of WYT, large language model (LLM) assistants were used in a limited and deliberate capacity as a reference and implementation guide. Their use is acknowledged transparently below.

**What AI tools were used for:**
- Providing API usage examples for unfamiliar Android SDK methods (e.g., `WifiManager` scan throttling workarounds on Android 9+, ONNX Runtime session initialisation on Android)
- Suggesting initial code structure for standard design patterns (e.g., Android foreground service lifecycle, observer/listener pattern for multi-component broadcasting)
- Answering targeted questions about mathematical concepts encountered in the literature (e.g., low-variance resampling, effective sample size in Particle Filters)

**What AI tools were NOT used for:**
- AI tools did not design the system architecture. The four-layer fusion pipeline, the P0–P3 correction hierarchy, the decision to use xLSTM as a displacement corrector rather than an absolute position source, and all algorithmic design choices were made independently by the developer based on the literature review.
- AI tools did not write the thesis. All written chapters were composed by the developer; AI assistance was limited to grammar and clarity suggestions consistent with university guidelines.
- AI tools did not validate correctness. All AI-suggested code was reviewed, tested against the live application, and verified against official Android and ONNX Runtime documentation before use. No AI output was accepted without full comprehension.

All architectural decisions, bug identification and resolution, algorithm selection, and academic claims made in this project are the sole work of the developer. AI tools served exclusively as an accessible reference layer — analogous to consulting Stack Overflow or a senior developer — not as an autonomous contributor to the codebase or academic content.

This acknowledgement aligns with the [BCS Code of Conduct](https://www.bcs.org/membership-and-registrations/become-a-member/bcs-code-of-conduct/) requirement to maintain professional competence and integrity, and with the University of Westminster's guidelines on the responsible use of generative AI in academic work.

---

## Licence

This project was developed as a Final Year Project for academic purposes. All source code is made available for academic review. Third-party datasets (UJIIndoorLoc, Opportunity) are subject to their respective licences and must be obtained independently from the sources linked above.
