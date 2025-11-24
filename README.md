<div align="center">
🦅 GUACAMAYA
Automated Detection and Counting System for African Wildlife
Deep Learning Architecture Optimization for Automated Wildlife Detection and Counting in High-Resolution Aerial Surveys
Show Image
Show Image
Show Image
Show Image
Show Image
Master's Thesis Project
Master in Artificial Intelligence (MAIA)
Universidad de los Andes, Bogotá, Colombia
2025
🚀 Live Demo | 📂 Repository | 📄 Paper
</div>

👥 Authors
<table>
<tr>
<td align="center"><b>Inmaculada Concepción Rondón</b><sup>*</sup><br><sub>Project Lead & ML Engineer</sub></td>
<td align="center"><b>Jorge Mario Guaquetá</b><br><sub>Data Scientist & ML Engineer</sub></td>
<td align="center"><b>Daniel Santiago Trujillo</b><br><sub>Backend Developer & DevOps</sub></td>
<td align="center"><b>Daniela Alexandra Ortiz Santacruz</b><br><sub>Frontend Developer & UX Designer</sub></td>
</tr>
</table>
<sup>*</sup>Corresponding author: mackierondon1@gmail.com
Affiliation: Centro SINFONÍA, Universidad de los Andes, Carrera 1 No. 18A-12, Bogotá 111711, Colombia

📋 Table of Contents

Abstract
Key Results
System Architecture
Methodology
Performance Analysis
Usage Guide
Deployment
Technical Specifications
Citation


Abstract
Aerial wildlife surveys are essential for population monitoring in extensive ecosystems, but manual counting methods present critical limitations including visual fatigue, inter-observer variability (up to 40%), and prohibitive processing costs (40-50 person-hours per 1,000 images).
This work presents GUACAMAYA, an automated wildlife detection and counting system for African fauna that prioritizes data quality over architectural complexity.
🎯 Key Contributions
<table>
<tr>
<td width="25%" align="center">
<img src="https://img.shields.io/badge/mAP@0.5-61.4%25-success?style=flat-square" />
<br><b>Detection Accuracy</b>
</td>
<td width="25%" align="center">
<img src="https://img.shields.io/badge/F1--Score-59.2%25-blue?style=flat-square" />
<br><b>Overall Performance</b>
</td>
<td width="25%" align="center">
<img src="https://img.shields.io/badge/Baseline-80.4%25-orange?style=flat-square" />
<br><b>vs. HerdNet</b>
</td>
<td width="25%" align="center">
<img src="https://img.shields.io/badge/Speed-3×_Faster-red?style=flat-square" />
<br><b>Computational Efficiency</b>
</td>
</tr>
</table>

✅ Robust data engineering pipeline correcting critical indexation errors (1-6 → 0-5)
✅ YOLO11s implementation achieving 61.4% mAP@0.5 and 59.2% F1-Score
✅ 80.4% baseline performance with 3× computational efficiency
✅ Full-stack deployment on AWS EC2 + Streamlit Cloud

Keywords: wildlife detection, YOLO, deep learning, data engineering, conservation, computer vision

🏆 Key Results
Performance Comparison
mermaidgraph LR
    A[HerdNet Baseline] -->|73.6% F1-Score| B[Reference Performance]
    C[GUACAMAYA YOLO11s] -->|59.2% F1-Score| D[80.4% of Baseline]
    C -->|3× Faster| E[Superior Efficiency]
    C -->|61.4% mAP@0.5| F[High Precision]
    
    style A fill:#ff6b6b
    style C fill:#4ecdc4
    style D fill:#95e1d3
    style E fill:#f38181
    style F fill:#aa96da
Critical Discovery
<div align="center">
```mermaid
graph TD
    A[Dataset with Incorrect Labels] -->|Indexation 1-6| B[Model Training]
    B --> C{mAP@0.5}
    C -->|Initial Result| D[0.0% - Complete Failure]
E[Data Engineering Pipeline] -->|Correction 1-6 → 0-5| F[Fixed Dataset]
F --> G[Model Retraining]
G --> H{mAP@0.5}
H -->|Final Result| I[61.4% - Functional System]

D -.->|Data Quality > Architecture| I

style D fill:#ff4757,color:#fff
style I fill:#26de81,color:#fff
style E fill:#fed330

**Impact:** +61.4 percentage points improvement through data correction alone

</div>

### Per-Species Performance

<div align="center">

| 🦁 Species | n | mAP@0.5 | Precision | Recall | F1-Score | Difficulty |
|:---|---:|:---:|:---:|:---:|:---:|:---:|
| 🐃 **Buffalo** | 369 | <span style="color:green">**83.1%**</span> | 85.7% | 64.8% | 73.8% | ⭐ Easy |
| 🐘 **Elephant** | 102 | <span style="color:green">**80.3%**</span> | 62.2% | 78.4% | 69.4% | ⭐ Easy |
| 🦌 **Kudu** | 161 | <span style="color:blue">**76.6%**</span> | 58.5% | 88.2% | 70.3% | ⭐⭐ Moderate |
| 🦌 **Waterbuck** | 39 | <span style="color:orange">40.2%</span> | 52.8% | 38.5% | 44.5% | ⭐⭐⭐ Difficult |
| 🐗 **Warthog** | 43 | <span style="color:red">28.9%</span> | 30.4% | 34.9% | 32.5% | ⭐⭐⭐⭐ Very Difficult |

</div>

---

## 🏗️ System Architecture

### Infrastructure Overview
```mermaid
graph TB
    subgraph User["👤 User Interface"]
        UI[Web Browser]
    end
    
    subgraph Frontend["🎨 Frontend - Streamlit Cloud (PaaS)"]
        ST[Streamlit App<br/>streamlit_app.py]
        VIZ[Plotly Visualizations]
    end
    
    subgraph Backend["⚙️ Backend - AWS EC2 + Docker"]
        API[Flask REST API<br/>Port 8000]
        MODEL[Model Inference<br/>YOLO11s / HerdNet]
        DB[(SQLite Database<br/>Predictions Tracking)]
    end
    
    subgraph Storage["💾 Model Storage"]
        GD[Google Drive<br/>Team Storage]
    end
    
    UI -->|HTTPS| ST
    ST -->|HTTP POST| API
    API --> MODEL
    MODEL --> DB
    API -->|JSON + Images| ST
    ST -->|Render Results| UI
    GD -.->|Download Models| MODEL
    
    style UI fill:#e3f2fd
    style ST fill:#fff3e0
    style API fill:#f3e5f5
    style MODEL fill:#e8f5e9
    style DB fill:#fce4ec
    style GD fill:#fff9c4
Processing Workflow
mermaidsequenceDiagram
    participant U as 👤 User
    participant F as 🎨 Frontend
    participant A as ⚙️ API Backend
    participant M as 🤖 YOLO11s
    participant D as 💾 Database
    
    U->>F: Upload Image(s)
    F->>F: Validate Format
    F->>A: HTTP POST /api/analyze
    A->>M: Load Model
    M->>M: Inference (2-3s)
    M->>A: Detections + Bounding Boxes
    A->>D: Log Results (Task ID)
    D->>A: Confirmation
    A->>F: JSON + Annotated Images
    F->>F: Generate Visualizations
    F->>U: Display Results + Download
    
    Note over M: GPU: Tesla T4<br/>Resolution: 2048px<br/>Confidence: 0.25
Technology Stack
<div align="center">
LayerTechnologyPurpose🎨 FrontendStreamlit Cloud (PaaS)Web interface, visualization⚙️ BackendAWS EC2 + DockerAPI server, model inference🤖 ML FrameworkPyTorch + Ultralytics YOLODeep learning engine💾 DatabaseSQLitePredictions tracking📊 VisualizationPlotlyInteractive charts🔐 APIFlask RESTBackend communication
</div>

🔬 Methodology
Dataset Characteristics
mermaidpie title Species Distribution in Test Set (n=714)
    "Buffalo" : 369
    "Kudu" : 161
    "Elephant" : 102
    "Warthog" : 43
    "Waterbuck" : 39
<div align="center">
HerdNet African Wildlife Dataset (Delplanque et al., 2022)
AttributeSpecification📸 Total Images~2,000 aerial photographs📐 Resolution5000×4000 pixels (20 MP)🎨 FormatJPEG, 24-bit RGB📏 GSD3-5 cm/pixel✈️ Capture AngleOblique (30-45° from nadir)🛩️ Flight Altitude100-150 meters AGL🌍 LocationEnnedi Reserve, Chad🏷️ Total Annotations6,962 instances across 6 species
</div>
Data Engineering Pipeline
mermaidflowchart LR
    A[Raw VOC Format<br/>Coordinates: x1,y1,x2,y2<br/>Classes: 1-6] --> B{Error Detection}
    B -->|Indexation Error| C[Correction Pipeline]
    C --> D[Reindex Classes<br/>1-6 → 0-5]
    D --> E[Convert to YOLO<br/>xc,yc,w,h normalized]
    E --> F[Validate Format]
    F --> G[Training Dataset<br/>Ready for YOLO11s]
    
    style A fill:#ff6b6b,color:#fff
    style C fill:#fed330
    style G fill:#26de81,color:#fff
Critical Error Correction
pythondef correct_annotation_file(filepath):
    """
    Corrects class indexation from 1-6 to 0-5 (YOLO standard)
    
    Impact: 0% → 61.4% mAP@0.5
    Files corrected: 400 annotation files
    """
    corrected_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0]) - 1  # Critical correction
                parts[0] = str(class_id)
                corrected_lines.append(' '.join(parts))
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(corrected_lines))
Model Architecture: YOLO11s
mermaidgraph TB
    subgraph Input["Input Layer"]
        I[Image: 2048×2048×3]
    end
    
    subgraph Backbone["Backbone: CSPDarknet"]
        B1[Conv Layer 1]
        B2[Conv Layer 2]
        B3[Conv Layer 3]
        B4[Conv Layer 4]
    end
    
    subgraph Neck["Neck: PANet"]
        N1[Feature Pyramid]
        N2[Path Aggregation]
    end
    
    subgraph Head["Detection Head"]
        H1[Large Objects]
        H2[Medium Objects]
        H3[Small Objects]
    end
    
    subgraph Output["Output"]
        O[Bounding Boxes<br/>+ Class Probabilities<br/>+ Confidence Scores]
    end
    
    I --> B1 --> B2 --> B3 --> B4
    B4 --> N1 --> N2
    N2 --> H1 & H2 & H3
    H1 & H2 & H3 --> O
    
    style I fill:#e3f2fd
    style B4 fill:#fff3e0
    style N2 fill:#f3e5f5
    style O fill:#e8f5e9
Configuration:

Input Resolution: 2048×2048 pixels
Training: 30 epochs, batch size=4, SGD optimizer (lr=0.01, momentum=0.937)
Hardware: Google Colab Pro, Tesla T4 GPU (16GB VRAM)
Framework: Ultralytics YOLO v8.3.229


📊 Performance Analysis
Scenario-Based Evaluation
mermaidgraph LR
    A[Open Savanna] -->|100% Recall| B[23/23 Detected]
    C[Multi-Species] -->|100% Recall| D[18/18 Detected]
    E[Dense Vegetation] -->|77.8% Recall| F[7/9 Detected]
    G[Cryptic Species] -->|33.3% Recall| H[4/12 Detected]
    
    style B fill:#26de81,color:#fff
    style D fill:#26de81,color:#fff
    style F fill:#fed330
    style H fill:#ff4757,color:#fff
<div align="center">
🎯 ScenarioComplexityAnimalsDetectedRecallPrecisionAvg.ConfidenceOpen savanna (Buffalo)⭐23/23100%95.7%84.2%Multi-species waterhole⭐⭐18/18100%94.4%78.6%Dense vegetation⭐⭐⭐7/977.8%87.5%64.3%Dense herd (>50 animals)⭐⭐⭐⭐58/6786.6%89.2%71.2%Cryptic species (Warthog)⭐⭐⭐⭐4/1233.3%50.0%38.7%
</div>
Computational Efficiency
mermaidgantt
    title Inference Time Comparison (per image)
    dateFormat X
    axisFormat %Ls
    
    section HerdNet
    HerdNet Processing    :0, 3000
    
    section GUACAMAYA
    GUACAMAYA Processing  :0, 1000
<div align="center">
MetricGUACAMAYA<br/>(YOLO11s)HerdNet<br/>BaselineImprovement⚡ Inference Time (GPU)0.5-1 s1.5-3 s3× faster💾 VRAM Usage2-3 GB4-5 GB40% ↓⚙️ CPU Usage30-50%60-80%38% ↓💿 Model Size180 MB85 MB—
</div>
Batch Processing Case Study
Scenario: Complete census of Ennedi Reserve North Sector
<table align="center">
<tr>
<td align="center" width="50%">
<b>📊 Processing Metrics</b><br/>
45 images | 15 km²<br/>
⏱️ 3 min 24 s total<br/>
🚀 13.2 images/min<br/>
✅ 487 animals detected
</td>
<td align="center" width="50%">
<b>🦁 Species Distribution</b><br/>
🐃 Buffalo: 198 (40.7%)<br/>
🐘 Elephant: 134 (27.5%)<br/>
🦌 Kudu: 89 (18.3%)<br/>
🦌 Waterbuck: 42 (8.6%)<br/>
🐗 Warthog: 24 (4.9%)
</td>
</tr>
</table>

💻 Usage Guide
🌐 Access
Production URL: https://guacamaya-app.streamlit.app
📸 Workflow
mermaidstateDiagram-v2
    [*] --> Upload: Access Application
    Upload --> Configure: Image(s) Uploaded
    Configure --> Execute: Parameters Set
    Execute --> Visualize: Processing Complete
    Visualize --> Export: Review Results
    Export --> [*]: Download Results
    
    Upload: 📤 Upload Images<br/>(.jpg, .png, .zip)
    Configure: ⚙️ Configure<br/>Model & Parameters
    Execute: 🔍 Execute Analysis<br/>(2-5s per image)
    Visualize: 📊 Visualize Results<br/>Charts + Tables
    Export: 💾 Export<br/>Images, CSV, PDF
Step-by-Step Instructions
<details>
<summary><b>Step 1: Image Upload</b> (Click to expand)</summary>

Single image: .jpg, .png (max 50MB)
Batch processing: .zip file (max 100 images)
Recommended format: 5000×4000 pixels (20MP)

</details>
<details>
<summary><b>Step 2: Parameter Configuration</b></summary>
ParameterOptionsDefaultDescriptionModelYOLO11s / HerdNetYOLO11sSpeed vs accuracy trade-offConfidence0.1 - 0.90.25Detection sensitivityIoU Threshold0.3 - 0.90.45Duplicate suppressionImage Size640 / 1280 / 20482048Resolution vs speed
</details>
<details>
<summary><b>Step 3: Execute Analysis</b></summary>

System generates unique task ID (e.g., task_20250615_143022)
Processing time: 2-5 seconds per image (GPU)
Real-time progress bar displayed

</details>
<details>
<summary><b>Step 4: Results Visualization</b></summary>
Output includes:

📊 Executive summary (detections, species, processing time)
📈 Species distribution (bar chart + pie chart)
🖼️ Annotated images with color-coded bounding boxes
📋 Detailed detection table (species, confidence, coordinates)

</details>
<details>
<summary><b>Step 5: Export Results</b></summary>
Available formats:

📥 Annotated images (.jpg)
📥 Detection table (.csv)
📥 Executive report (.pdf)
📥 Complete bundle (.zip)

</details>

🚀 Deployment
Local Development
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

# Launch backend (Terminal 1)
python app/backend/api_server.py

# Launch frontend (Terminal 2)
streamlit run app/streamlit_app.py
AWS EC2 Production
bash# Connect to EC2
ssh -i ~/.ssh/guacamaya-key.pem ubuntu@<EC2_PUBLIC_IP>

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Clone and configure
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal
cp config/.env.example config/.env

# Build and run
docker build -t guacamaya-backend:latest -f docker/Dockerfile .
docker run -d --name guacamaya-backend -p 8000:8000 \
  -v $(pwd)/modelos:/app/modelos \
  --env-file config/.env \
  --restart always \
  guacamaya-backend:latest

# Verify
curl http://localhost:8000/health

🔧 Technical Specifications
Dependencies
<div align="center">
CategoryKey LibrariesVersion🧠 Deep LearningPyTorch, Ultralytics YOLO2.0.1, 8.3.229🌐 Web FrameworkStreamlit, Flask1.28.0, 3.0.0🖼️ Computer VisionOpenCV, Pillow4.8.1, 10.1.0📊 Data ProcessingNumPy, Pandas1.24.3, 2.1.0📈 VisualizationPlotly, Matplotlib5.17.0, 3.7.2
</div>
Installation:
bashpip install -r requirements.txt
Hardware Requirements
<div align="center">
ComponentMinimumRecommendedNotesCPUIntel i5Intel i7—RAM8GB16GBRequired for batch processingGPUNoneNVIDIA (4GB+ VRAM)5-10× speedupStorage2GB10GBModels + datasetsNetwork10 Mbps100 MbpsFor cloud deployment
</div>

🎯 Limitations and Future Work
Current Limitations
mermaidmindmap
  root((Limitations))
    Species Coverage
      Warthog: 28.9% mAP
      Cryptic species challenging
    Resolution
      Training: 2048px
      Native: 5000px gap
    Occlusion
      >50% reduces detection
      20-25% recall drop
    Training
      Only 30 epochs
      Time constrained
    Data Imbalance
      Waterbuck: 39 samples
      Insufficient for learning
Proposed Improvements
<table>
<tr>
<td width="33%" valign="top">
🔬 Technical

Advanced data balancing
Adaptive patching
Extended training (50-100 epochs)
Attention mechanisms

</td>
<td width="33%" valign="top">
⚡ Deployment

Model quantization (INT8)
Mobile deployment (ONNX)
Multi-GPU processing
Cloud auto-scaling

</td>
<td width="33%" valign="top">
🧪 Validation

Field trials
Inter-annotator studies
Longitudinal tracking
Cross-ecosystem testing

</td>
</tr>
</table>

📚 References
Scientific Publications
[1] Delplanque, A., et al. (2022). Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks. Remote Sensing in Ecology and Conservation, 8(2), 166–179.
[2] Delplanque, A., et al. (2023). From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning? ISPRS Journal of Photogrammetry and Remote Sensing, 197, 167-180.
[3] Kellenberger, B., et al. (2018). Detecting mammals in UAV images: Best practices to address a substantially imbalanced dataset with deep learning. Remote Sensing of Environment, 216, 139–153.
[4] Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. IEEE CVPR, 779-788.
[5] Jocher, G., et al. (2023). Ultralytics YOLO. GitHub repository. https://github.com/ultralytics/ultralytics

🙏 Acknowledgments
<table>
<tr>
<td width="33%" align="center">
<b>🏢 Technical Support</b><br/>
Microsoft AI for Good Lab<br/>
Centro SINFONÍA, Uniandes<br/>
AWS Educate
</td>
<td width="33%" align="center">
<b>🌿 Conservation Expertise</b><br/>
Instituto Sinchi<br/>
Instituto Alexander<br/>von Humboldt
</td>
<td width="33%" align="center">
<b>💻 Open Source</b><br/>
Ultralytics Team<br/>
PyTorch Community<br/>
Streamlit Team
</td>
</tr>
</table>

📞 Contact
<div align="center">
Corresponding Author: Inmaculada Concepción Rondón
📧 mackierondon1@gmail.com
Show Image
Show Image
Show Image
</div>

📝 Citation
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

📄 License
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

🦅 Built with dedication by Grupo 12
Master in Artificial Intelligence (MAIA)
Universidad de los Andes, Bogotá, Colombia
2025

"Data quality is more determinant than algorithmic sophistication in deep learning applications for conservation."

</div>
