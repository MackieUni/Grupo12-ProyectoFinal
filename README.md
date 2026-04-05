# GUACAMAYA: Automated Wildlife Detection System

**Data Quality Over Model Complexity: YOLO11-Based Wildlife Detection Achieving 80% Baseline Performance with 60% Fewer Parameters**

## Defense & Intelligence Applications

The computer vision techniques developed in GUACAMAYA: 
ultra-high resolution aerial imagery analysis, 
multi-class object detection at scale, edge-optimized 
inference with 62% parameter reduction, and automated 
population counting — translate directly to 
Intelligence, Surveillance, and Reconnaissance (ISR) 
applications in defense environments. This work 
demonstrates production-grade computer vision 
engineering under real-world constraints.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLO v11s](https://img.shields.io/badge/YOLO-v11s-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

---

## Overview

**GUACAMAYA** is an automated detection and counting system for African mammals in ultra-high resolution aerial imagery (20MP), developed at Universidad de los Andes in collaboration with Microsoft AI for Good Lab and Centro SINFONÍA.

The system implements a YOLO11s architecture optimized for wildlife conservation applications, achieving **61.4% mAP@0.5** and **59.2% F1-Score** — equivalent to 80.4% of the HerdNet baseline performance while using 62% fewer parameters (9.4M vs ~25M) and providing 3× computational efficiency.

**Key Finding**: Data quality engineering contributed +61.4 percentage points improvement, empirically demonstrating that dataset correction exceeds architectural optimization gains by an order of magnitude in conservation AI applications.

### Species Detected

| Species | Test Instances | mAP@0.5 | Detection Difficulty |
|---------|---------------|---------|---------------------|
| Buffalo | 369 | 83.1% | Low |
| Elephant | 102 | 80.3% | Low |
| Kob | 161 | 76.6% | Moderate |
| Topi | — | — | Limited data |
| Warthog | 43 | 28.9% | High (cryptic) |
| Waterbuck | 39 | 40.2% | High (minority class) |

---

## Quick Start

### Live Demo

**Application**: [https://guacamaya-app.streamlit.app](https://guacamaya-app.streamlit.app)

### Local Installation

```bash
# Clone repository
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py

# Run application
streamlit run app/streamlit_app.py
```

### Docker Deployment

```bash
docker build -t guacamaya-backend:latest -f docker/Dockerfile .
docker run -d -p 8000:8000 --name guacamaya-backend guacamaya-backend:latest
```

---

## Architecture

```
┌─────────────────────┐     HTTP      ┌─────────────────────┐
│  Frontend           │ ◄──────────► │  Backend            │
│  (Streamlit Cloud)  │    REST API  │  (AWS EC2 + Docker) │
│                     │              │                     │
│  • Image upload     │              │  • YOLO11s model    │
│  • Parameter config │              │  • HerdNet baseline │
│  • Visualization    │              │  • SQLite tracking  │
│  • Report export    │              │  • Batch processing │
└─────────────────────┘              └─────────────────────┘
```

### Technical Stack

| Component | Technology |
|-----------|------------|
| Detection Model | YOLO11s (Ultralytics) |
| Backend API | Flask REST (Port 8000) |
| Frontend | Streamlit |
| Infrastructure | AWS EC2 (t3.large) |
| Database | SQLite |
| Containerization | Docker |

---

## Results

### Performance Metrics

| Metric | GUACAMAYA | HerdNet (Baseline) | Comparison |
|--------|-----------|-------------------|------------|
| mAP@0.5 | 61.4% | 76.4% | 80.4% |
| F1-Score | 59.2% | 73.6% | 80.4% |
| Parameters | 9.4M | ~25M | 62% fewer |
| Inference Speed | ~2-3s | ~6-9s | 3× faster |

### Critical Finding: Data Engineering Impact

The discovery and correction of a class indexation error (labels 1-6 → 0-5) across 1,297 annotation files resulted in:

| Condition | mAP@0.5 |
|-----------|---------|
| Before correction | 0.0% |
| After correction | 61.4% |
| **Improvement** | **+61.4 pp** |

This gain exceeds typical architectural optimizations (2-5 pp) by an order of magnitude, empirically establishing that **data quality is the primary bottleneck** in wildlife detection applications.

### Negative Results (Documented for Scientific Transparency)

Two fine-tuning experiments resulted in performance degradation:

| Experiment | Result | Cause |
|------------|--------|-------|
| Fine-tuning Exp. 1 | -13.2 pp | Catastrophic forgetting |
| Fine-tuning Exp. 2 | -9.5 pp | Hyperparameter sensitivity |

---

## Repository Structure

```
Grupo12-ProyectoFinal/
├── app/                    # Web application
│   ├── streamlit_app.py    # Frontend interface
│   └── backend/            # REST API server
├── notebooks/              # Jupyter notebooks (experiments)
├── modelos/                # Trained model weights
├── datos/                  # Sample images and annotations
├── config/                 # Configuration files
├── scripts/                # Utility scripts
├── docker/                 # Docker configuration
├── docs/                   # Documentation and paper
└── tests/                  # Automated tests
```

---

## Dataset

This project uses the **HerdNet African Wildlife Dataset** (Delplanque et al., 2022):

| Property | Value |
|----------|-------|
| Total Images | ~2,000 |
| Resolution | 5000 × 4000 px (20MP) |
| Total Annotations | 6,962 instances |
| Location | Ennedi Reserve, Chad |
| Train/Val/Test Split | 70/10/20 |

**Access**: Sample images included in `datos/sample_images/`. Full dataset available from [Delplanque et al., 2022](https://doi.org/10.1002/rse2.234).

---

## Usage

### Web Interface

1. Navigate to [https://guacamaya-app.streamlit.app](https://guacamaya-app.streamlit.app)
2. Upload aerial image(s) (JPEG/PNG, max 50MB)
3. Configure detection parameters:
   - **Model**: YOLO11s (recommended) or HerdNet
   - **Confidence threshold**: 0.25 (default)
   - **IoU threshold**: 0.45 (default)
   - **Image size**: 2048px (default)
4. Click "Run Analysis"
5. Download results (CSV, annotated images, PDF report)

### API Endpoint

```bash
curl -X POST http://api-url:8000/api/analyze \
  -F "file=@image.jpg" \
  -F "model=yolo11s" \
  -F "confidence=0.25"
```

### Programmatic Usage

```python
from ultralytics import YOLO

model = YOLO("modelos/yolo11s_best.pt")
results = model.predict("image.jpg", conf=0.25, iou=0.45, imgsz=2048)
```

---

## Requirements

### Python Dependencies

```
torch>=2.0.1
ultralytics>=8.3.229
streamlit>=1.28.0
flask>=3.0.0
opencv-python>=4.8.1
pandas>=2.1.0
plotly>=5.17.0
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| GPU | — | CUDA-compatible |
| Storage | 500 MB | 2 GB |

**Performance**: CPU inference ~3-5s/image; GPU (Tesla T4) ~0.5-1s/image.

---

## Citation

```bibtex
@mastersthesis{guacamaya2025,
  title={Data Quality Over Model Complexity: YOLO11-Based Wildlife Detection 
         Achieving 80\% Baseline Performance with 60\% Fewer Parameters},
  author={Rondón, Inmaculada Concepción and Guaquetá, Jorge Mario and 
          Trujillo, Daniel Santiago and Ortiz Santacruz, Daniela Alexandra},
  year={2025},
  school={Universidad de los Andes},
  department={Master's Program in Artificial Intelligence (MAIA)},
  address={Bogotá, Colombia}
}
```

---

## Authors

| Name | Role | Contact |
|------|------|---------|
| **Inmaculada C. Rondón*** | Project Lead, ML Engineer | ic.rondon@uniandes.edu.co |
| Jorge Mario Guaquetá | Data Scientist | jm.guaqueta@uniandes.edu.co |
| Daniel Santiago Trujillo | Backend Developer | ds.trujillo@uniandes.edu.co |
| Daniela A. Ortiz Santacruz | Frontend Developer | da.ortiz@uniandes.edu.co |

*Corresponding author

**Affiliation**: Centro SINFONÍA, Universidad de los Andes, Bogotá 111711, Colombia

---

## Acknowledgments

- **Microsoft AI for Good Lab** — Technical guidance and computational support
- **Centro SINFONÍA, Universidad de los Andes** — Research infrastructure
- **AWS Educate** — Cloud computing credits
- **Ultralytics Team** — YOLO11 open-source implementation
- **Delplanque et al.** — HerdNet dataset
- **Professor Juan Carlos Olarte** — Project supervision

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Master's Program in Artificial Intelligence (MAIA)**  
**Universidad de los Andes | 2025**

*"Before adding layers to your neural network, add layers of validation to your data."*

</div>
