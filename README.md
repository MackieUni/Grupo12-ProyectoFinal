GUACAMAYA: Automated Detection and Counting System for African Wildlife
Deep Learning Architecture Optimization for Automated Wildlife Detection and Counting in High-Resolution Aerial Surveys
<div align="center">
Show Image
Show Image
Show Image
Show Image
Show Image
Master's Thesis Project
Master in Artificial Intelligence (MAIA)
Universidad de los Andes, Bogotá, Colombia
2025
</div>

Authors
Inmaculada Concepción Rondón<sup>a,*</sup> · Jorge Mario Guaquetá<sup>a</sup> · Daniel Santiago Trujillo<sup>a</sup> · Daniela Alexandra Ortiz Santacruz<sup>a</sup>
<sup>a</sup>Centro SINFONÍA, Universidad de los Andes, Carrera 1 No. 18A-12, Bogotá 111711, Colombia
<sup>*</sup>Corresponding author: mackierondon1@gmail.com

Table of Contents

Abstract
1. Introduction
2. Methodology
3. Results
4. System Architecture
5. Usage Instructions
6. Technical Specifications
7. Deployment Guide
8. Performance Benchmarks
9. Limitations and Future Work
10. References


Abstract
Aerial wildlife surveys are essential for population monitoring in extensive ecosystems, but manual counting methods present critical limitations including visual fatigue, inter-observer variability (up to 40%), and prohibitive processing costs (40-50 person-hours per 1,000 images). This work presents GUACAMAYA, an automated wildlife detection and counting system for African fauna that prioritizes data quality over architectural complexity.
Key Contributions:

Development of robust data engineering pipeline correcting critical indexation errors (1-6 → 0-5)
Implementation of YOLO11s achieving 61.4% mAP@0.5 and 59.2% F1-Score
Demonstration of 80.4% baseline performance (HerdNet) with 3× computational efficiency
Full-stack deployment on AWS EC2 + Streamlit Cloud infrastructure

Keywords: wildlife detection, YOLO, deep learning, data engineering, conservation, computer vision

1. Introduction
1.1 Problem Statement
Traditional aerial wildlife surveys rely on manual counting by human observers from aircraft, introducing systematic errors:
LimitationImpactVisual fatigue>30% accuracy reduction after 2 hoursTurbulence effectsInconsistent observation conditionsInter-observer variabilityUp to 40% discrepancy between observersProcessing time40-50 person-hours per 1,000 imagesScalabilityCannot process systematic survey volumes
1.2 Research Objectives

Primary: Develop automated wildlife detection system for ultra-high resolution (20MP) aerial imagery
Secondary: Achieve ≥80% of baseline HerdNet performance with improved computational efficiency
Tertiary: Demonstrate practical deployment viability for conservation applications

1.3 Project Evolution
PhaseInitial PlanImplemented SolutionArchitectureHerdNet vs YOLO comparisonYOLO11s optimizationStrategyMulti-architecture evaluationData quality prioritizationChallengePatch management for 20MPCritical annotation correction (400 files)Result—0% → 61.4% mAP transformation
Critical Discovery: Data indexation error (1-6 instead of 0-5) caused complete model failure. Correction pipeline became primary contribution.

2. Methodology
2.1 Dataset Characteristics
HerdNet African Wildlife Dataset (Delplanque et al., 2022)
AttributeSpecificationTotal images~2,000 aerial photographsResolution5000×4000 pixels (20 megapixels)FormatJPEG, 24-bit RGBGSD3-5 cm/pixelCapture angleOblique (30-45° from nadir)Flight altitude100-150 meters AGLLocationEnnedi Reserve, ChadTotal annotations6,962 instances
Species Distribution (Test Set, n=714):
SpeciesCodeInstances% TotalDifficultyBuffalo036953.0%EasyKudu216123.1%ModerateElephant110214.6%EasyWarthog4436.2%Very difficultWaterbuck3395.6%Difficult
2.2 Data Engineering Pipeline
Critical Error Correction:
pythondef correct_annotation_file(filepath):
    """Corrects class indexation from 1-6 to 0-5 (YOLO standard)"""
    corrected_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0]) - 1  # Decrement by 1
                parts[0] = str(class_id)
                corrected_lines.append(' '.join(parts))
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(corrected_lines))
Format Conversion (VOC → YOLO):
pythondef voc_to_yolo(x1, y1, x2, y2, img_w=5000, img_h=4000):
    """Converts bounding box from VOC to YOLO normalized format"""
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h
2.3 Model Architecture
YOLO11s Configuration:

Input resolution: 2048×2048 pixels
Backbone: CSPDarknet with speed optimizations
Neck: PANet for multi-scale feature fusion
Head: Anchor-based detector with 3 detection scales
Training: 30 epochs, batch size=4, SGD optimizer (lr=0.01, momentum=0.937)
Hardware: Google Colab Pro, Tesla T4 GPU (16GB VRAM)
Framework: Ultralytics YOLO v8.3.229


3. Results
3.1 Overall Performance
MetricValuevs. HerdNet BaselinemAP@0.561.4%83.4%mAP@0.5:0.9529.8%—F1-Score59.2%80.4%Precision57.7%78.4%Recall60.8%82.6%Inference speed~2-3s/image3× faster
3.2 Per-Species Performance
SpeciesnmAP@0.5PrecisionRecallF1-ScoreBuffalo36983.1%85.7%64.8%73.8%Elephant10280.3%62.2%78.4%69.4%Kudu16176.6%58.5%88.2%70.3%Waterbuck3940.2%52.8%38.5%44.5%Warthog4328.9%30.4%34.9%32.5%
Analysis: Large-bodied species (Buffalo, Elephant) achieve >80% mAP due to high visual contrast. Cryptic species (Warthog) exhibit low performance (28.9%) attributable to natural camouflage and low-posture morphology.
3.3 Impact of Data Correction
ConditionmAP@0.5StatusIncorrect labels (1-6)0.0%Non-functionalCorrected labels (0-5)61.4%FunctionalImprovement+61.4 pp—
This result validates data quality as the primary determinant of model performance in specialized deep learning applications.

4. System Architecture
4.1 Infrastructure Overview
┌─────────────────────────────────────────────────────────────┐
│                    GUACAMAYA SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐          ┌──────────────────┐            │
│  │   Frontend   │  HTTP    │     Backend      │            │
│  │  Streamlit   │ ◄──────► │   AWS EC2        │            │
│  │   (PaaS)     │  REST    │   Docker:8000    │            │
│  └──────────────┘          └──────────────────┘            │
│         │                          │                        │
│         │                          ▼                        │
│         │                  ┌──────────────┐                │
│         │                  │  YOLO11s     │                │
│         │                  │  HerdNet     │                │
│         │                  └──────────────┘                │
│         │                          │                        │
│         │                          ▼                        │
│         │                  ┌──────────────┐                │
│         └─────────────────►│   SQLite DB  │                │
│                            └──────────────┘                │
└─────────────────────────────────────────────────────────────┘
4.2 Backend Specifications
ComponentTechnologyInfrastructureAWS EC2 t3.large (2 vCPUs, 8GB RAM)ContainerizationDockerAPI FrameworkFlask REST APIPort8000DatabaseSQLite (predictions tracking)Model StorageGoogle Drive (team)Model FormatsYOLO11s (.pt, 180MB), HerdNet (.pt, 85MB)
4.3 Frontend Specifications
ComponentTechnologyPlatformStreamlit Cloud (PaaS)LanguagePython 3.10Main Filestreamlit_app.pyDeploymentOn-demand from GitHubVisualizationPlotly interactive chartsInterfaceWeb browser (cross-platform)
4.4 Processing Workflow
1. User uploads image(s) → Streamlit frontend
2. HTTP POST request → Backend API (port 8000)
3. Model loading → YOLO11s/HerdNet
4. Inference execution → Bounding box detection
5. Results logging → SQLite database
6. Response generation → JSON + annotated images
7. Visualization → Frontend rendering
4.5 Configurable Parameters
ParameterRangeDefaultEffectModelYOLO11s / HerdNetYOLO11sSpeed vs accuracy trade-offConfidence threshold0.1 - 0.90.25Detection sensitivityIoU threshold0.3 - 0.90.45Duplicate suppressionImage size640 / 1280 / 20482048Resolution vs speed

5. Usage Instructions
5.1 Access
Production URL: https://guacamaya-app.streamlit.app
5.2 Workflow
Step 1: Image Upload

Single image: .jpg, .png (max 50MB)
Batch processing: .zip file (max 100 images)
Recommended format: 5000×4000 pixels (20MP)

Step 2: Parameter Configuration

Select model: YOLO11s (recommended) or HerdNet
Set confidence threshold: 0.25 (default)
Set IoU threshold: 0.45 (default)
Choose image size: 2048px (default)

Step 3: Execute Analysis

System generates unique task ID (e.g., task_20250615_143022)
Processing time: 2-5 seconds per image (GPU)

Step 4: Results Visualization

Executive summary (detections, species, processing time)
Species distribution (bar chart + pie chart)
Annotated images with color-coded bounding boxes
Detailed detection table (species, confidence, coordinates)

Step 5: Export Results

Download annotated images (.jpg)
Download detection table (.csv)
Download executive report (.pdf)
Download complete bundle (.zip)


6. Technical Specifications
6.1 Dependencies
Core Framework:
Python 3.8-3.11 (recommended: 3.10.x)
torch==2.0.1
torchvision==0.15.2
ultralytics==8.3.229
Web Framework:
streamlit==1.28.0
flask==3.0.0
flask-cors==4.0.0
Computer Vision:
opencv-python==4.8.1.78
Pillow==10.1.0
Data Processing:
numpy==1.24.3
pandas==2.1.0
Visualization:
matplotlib==3.7.2
plotly==5.17.0
seaborn==0.12.2
Installation:
bashpip install -r requirements.txt
6.2 Hardware Requirements
ComponentMinimumRecommendedCPUIntel i5 / AMD Ryzen 5Intel i7 / AMD Ryzen 7RAM8GB16GBGPUNone (CPU mode)NVIDIA GPU (4GB+ VRAM)Storage2GB10GBNetwork10 Mbps100 Mbps
6.3 Performance Metrics
HardwareInference TimeThroughputCPU (Intel i7)3-5 s12-20 images/minGPU (Tesla T4)0.5-1 s60-120 images/minCold start10-15 s—

7. Deployment Guide
7.1 Local Development
bash# Clone repository
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Configure environment
cp config/.env.example config/.env
nano config/.env  # Edit configuration

# Launch backend
python app/backend/api_server.py

# Launch frontend (separate terminal)
streamlit run app/streamlit_app.py
7.2 AWS EC2 Production Deployment
bash# Connect to EC2 instance
ssh -i ~/.ssh/guacamaya-key.pem ubuntu@<EC2_PUBLIC_IP>

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal

# Configure environment
cp config/.env.example config/.env
nano config/.env

# Build Docker image
docker build -t guacamaya-backend:latest -f docker/Dockerfile .

# Run container
docker run -d \
  --name guacamaya-backend \
  -p 8000:8000 \
  -v $(pwd)/modelos:/app/modelos \
  -v $(pwd)/backend/database:/app/backend/database \
  --env-file config/.env \
  --restart always \
  guacamaya-backend:latest

# Verify deployment
curl http://localhost:8000/health
7.3 Streamlit Cloud Frontend

Navigate to streamlit.io/cloud
Click "New app"
Connect GitHub repository: MackieUni/Grupo12-ProyectoFinal
Set main file: app/streamlit_app.py
Configure secrets: API_BASE_URL = "http://<EC2_PUBLIC_IP>:8000"
Deploy application


8. Performance Benchmarks
8.1 Scenario-Based Evaluation
ScenarioComplexityn (actual)n (detected)RecallPrecisionAvg. ConfidenceOpen savanna (Buffalo)⭐2323100%95.7%84.2%Multi-species waterhole⭐⭐1818100%94.4%78.6%Dense vegetation⭐⭐⭐9777.8%87.5%64.3%Dense herd (>50 animals)⭐⭐⭐⭐675886.6%89.2%71.2%Cryptic species (Warthog)⭐⭐⭐⭐12433.3%50.0%38.7%
8.2 Computational Efficiency
MetricGUACAMAYA (YOLO11s)HerdNet BaselineImprovementInference time (GPU)0.5-1 s1.5-3 s3× fasterModel size180 MB85 MB—VRAM usage2-3 GB4-5 GB40% reductionCPU usage30-50%60-80%38% reduction
8.3 Batch Processing Case Study
Scenario: Complete census of Ennedi Reserve North Sector (45 images, 15 km²)
MetricValueTotal processing time3 min 24 sAverage per image4.5 sThroughput13.2 images/minTotal detections487 animalsSpecies identified5Average confidence76.4%
Species Distribution:

Buffalo: 198 (40.7%)
Elephant: 134 (27.5%)
Kudu: 89 (18.3%)
Waterbuck: 42 (8.6%)
Warthog: 24 (4.9%)


9. Limitations and Future Work
9.1 Current Limitations
CategoryLimitationImpactSpecies CoverageWarthog: 28.9% mAPLow detection rate for cryptic speciesResolutionTraining: 2048px vs Native: 5000pxPossible small animal misdetectionOcclusion>50% occlusion reduces detection20-25% recall reductionTraining30 epochs (time-constrained)Potential for further optimizationMinority ClassesWaterbuck: only 39 test samplesInsufficient data for robust learning
9.2 Proposed Future Work
Technical Improvements:

Advanced data balancing for minority species (augmentation, GANs)
Adaptive patching for native resolution (5000×4000) processing
Extended training (50-100 epochs) with learning rate scheduling
Attention mechanisms for cryptic species detection

Deployment Enhancements:

Real-time inference with model quantization (INT8)
Mobile deployment (ONNX/TensorFlow Lite)
Multi-GPU distributed processing
Cloud auto-scaling for batch workloads

Scientific Validation:

Field trials in African national parks
Inter-annotator agreement studies
Longitudinal population tracking
Cross-ecosystem generalization testing


10. References
10.1 Scientific Publications
[1] Delplanque, A., Foucher, S., Lejeune, P., Linchant, J., Théau, J. (2022). Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks. Remote Sensing in Ecology and Conservation, 8(2), 166–179. DOI: 10.1002/rse2.234
[2] Delplanque, A., Foucher, S., Théau, J., Bussière, E., Vermeulen, C., Lejeune, P. (2023). From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning? ISPRS Journal of Photogrammetry and Remote Sensing, 197, 167-180.
[3] Kellenberger, B., Marcos, D., Tuia, D. (2018). Detecting mammals in UAV images: Best practices to address a substantially imbalanced dataset with deep learning. Remote Sensing of Environment, 216, 139–153.
[4] Kellenberger, B., Marcos, D., Lobry, S., Tuia, D. (2019). Half a Percent of Labels is Enough: Efficient Animal Detection in UAV Imagery Using Deep CNNs and Active Learning. IEEE Transactions on Geoscience and Remote Sensing, 57(12), 9524–9533.
[5] Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779-788.
[6] Jocher, G., Chaurasia, A., Jing, Q. (2023). Ultralytics YOLO. GitHub repository. https://github.com/ultralytics/ultralytics
10.2 Dataset
Delplanque, A., et al. (2022). HerdNet African Wildlife Dataset. Available: https://doi.org/10.1002/rse2.234
10.3 Code and Resources
GitHub Repository: https://github.com/MackieUni/Grupo12-ProyectoFinal
Live Application: https://guacamaya-app.streamlit.app
Documentation: See docs/ directory in repository

Acknowledgments
This work was conducted as part of the Master in Artificial Intelligence (MAIA) program at Universidad de los Andes. We acknowledge the following institutions for their support:
Technical Support:

Microsoft AI for Good Lab (computational resources)
Centro SINFONÍA, Universidad de los Andes (academic infrastructure)
AWS Educate (cloud infrastructure credits)

Conservation Expertise:

Instituto Sinchi (biodiversity monitoring expertise)
Instituto Alexander von Humboldt (Colombian biodiversity applications)

Open Source Community:

Ultralytics Team (YOLO framework)
PyTorch Community (deep learning framework)
Streamlit Team (web application framework)


Contact
Corresponding Author:
Inmaculada Concepción Rondón
📧 mackierondon1@gmail.com
Project Repository:
🔗 https://github.com/MackieUni/Grupo12-ProyectoFinal
Issues & Feature Requests:
🐛 GitHub Issues
Live Demo:
🌐 https://guacamaya-app.streamlit.app

Citation
If you use this work in your research, please cite:
bibtex@mastersthesis{rondon2025guacamaya,
  title={GUACAMAYA: Automated Detection and Counting System for African Wildlife},
  author={Rond\'on, Inmaculada Concepci\'on and Guaquet\'a, Jorge Mario and 
          Trujillo, Daniel Santiago and Ortiz Santacruz, Daniela Alexandra},
  school={Universidad de los Andes},
  year={2025},
  type={Master's Thesis},
  department={Centro SINFONÍA},
  program={Master in Artificial Intelligence (MAIA)},
  note={System based on YOLO11s achieving 61.4\% mAP@0.5 (80.4\% of HerdNet baseline) 
        with 3$\times$ computational efficiency improvement}
}

License
MIT License

Copyright (c) 2025 Grupo 12 - Universidad de los Andes
Inmaculada Concepción Rondón, Jorge Mario Guaquetá, 
Daniel Santiago Trujillo, Daniela Alexandra Ortiz Santacruz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<div align="center">
⭐ Star this repository if you find it useful ⭐
Show Image

Built with dedication by Grupo 12
Master in Artificial Intelligence (MAIA)
Universidad de los Andes, Bogotá, Colombia
2025

"Data quality is more determinant than algorithmic sophistication in deep learning applications for conservation."
</div>
