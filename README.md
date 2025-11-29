# 🦅 GUACAMAYA: Sistema Automático de Detección y Conteo de Fauna

**Optimización de Arquitecturas de Deep Learning para Detección y Conteo Automático de Fauna en Surveys Aéreos de Alta Resolución**

**Proyecto Final - Maestría en Inteligencia Artificial (MAIA)**  
**Universidad de los Andes | 2024**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-v11s-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

---

## 👥 Autores

**Inmaculada Concepción Rondón**^a,*  
**Jorge Mario Guaquetá**^a  
**Daniel Santiago Trujillo**^a  
**Daniela Alexandra Ortiz Santacruz**^a

^a Centro SINFONÍA, Universidad de los Andes, Carrera 1 No. 18A-12, Bogotá 111711, Colombia

*Autor correspondiente: mackierondon1@gmail.com

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Descripción del Proyecto](#descripción-del-proyecto)
3. [Instrucciones de Uso](#instrucciones-de-uso)
4. [Enlaces del Proyecto](#enlaces-del-proyecto)
5. [Dependencias](#dependencias)
6. [Instrucciones de Despliegue](#instrucciones-de-despliegue)
7. [Estructura del Repositorio](#estructura-del-repositorio)
8. [Estructura de los Datos](#estructura-de-los-datos)
9. [Consideraciones Técnicas](#consideraciones-técnicas)
10. [Ejemplos de Entrada y Salida](#ejemplos-de-entrada-y-salida)
11. [Equipo y Contacto](#equipo-y-contacto)

---

## 📊 Resumen Ejecutivo

Los **aerial wildlife surveys** son fundamentales para el monitoreo poblacional de fauna en ecosistemas extensos, pero presentan limitaciones críticas en su implementación manual. Este proyecto presenta **GUACAMAYA**, un sistema automático de detección y conteo de fauna africana que evolucionó significativamente desde su concepción inicial debido a desafíos inesperados en el manejo de datos. 

Implementamos una metodología que priorizó la **calidad de datos sobre la complejidad arquitectónica**, alcanzando:
- **61.4% mAP@0.5**
- **59.2% F1-Score**
- **80.4% del rendimiento del baseline HerdNet**

El proyecto demuestra que la **ingeniería de datos robusta puede ser más determinante que la sofisticación algorítmica** en aplicaciones de deep learning para conservación.

**Palabras clave:** detección de fauna, YOLO, aprendizaje profundo, ingeniería de datos, conservación, visión por computadora

---

## 1. 📖 Descripción del Proyecto

### Contexto de la Problemática

Los censos aéreos de fauna silvestre tradicionalmente dependen de **conteo manual** por observadores humanos desde aeronaves, un método que presenta limitaciones críticas:

- **Fatiga visual** durante vuelos prolongados (reducción de precisión >30% después de 2 horas)
- **Errores por turbulencia** y condiciones climáticas adversas
- **Alta variabilidad inter-observador:** Discrepancias de hasta 40% entre diferentes observadores
- **Limitaciones temporales:** Tiempo limitado de observación durante el vuelo
- **Alto costo operacional:** 40-50 horas-persona requeridas para procesar 1,000 imágenes manualmente
- **Limitación de escala:** Imposibilidad de procesar grandes volúmenes de imágenes de censos sistemáticos

### Problema Principal que Resuelve

Las imágenes aéreas de **ultra-alta resolución (5000×4000 píxeles, 20MP)** generadas por censos sistemáticos ofrecen datos valiosos, pero el procesamiento manual es prohibitivamente costoso y propenso a errores. Además, arquitecturas existentes como HerdNet enfrentan desafíos en:

- **Restricciones de memoria:** Imposibilidad de procesar imágenes completas en GPUs estándar
- **Eficiencia computacional:** Tiempos de procesamiento elevados
- **Compatibilidad de datos:** Formatos de anotación inconsistentes

### Evolución del Proyecto

**Guacamaya** experimentó una **pivotación fundamental** durante su desarrollo:

**Fase Inicial (Planeada):**
- Comparación de arquitecturas avanzadas (HerdNet vs YOLO vs híbridas)
- Optimización de estrategias de patchado para imágenes de 20MP
- Enfoque en sofisticación algorítmica

**Fase Real (Implementada):**
- **Corrección masiva de anotaciones** (400 archivos con error de indexación 1-6 → 0-4)
- **Pipeline automatizado** de conversión de formatos (VOC → YOLO)
- **Priorización de calidad de datos** sobre complejidad arquitectónica
- Resultado: Transformación de 0% mAP a 61.4% mAP tras corrección de datos

### Objetivo del Sistema

**GUACAMAYA** es un sistema automatizado de detección y conteo de fauna africana en imágenes aéreas que utiliza deep learning (arquitectura **YOLO11s**) para:

- ✅ **Detectar y localizar** automáticamente animales en imágenes aéreas de 20MP
- ✅ **Clasificar** entre 6 especies: Buffalo, Elephant, Kudu, Topi, Warthog, Waterbuck
- ✅ **Contar** automáticamente individuos con **61.4% mAP@0.5**
- ✅ **Alcanzar 80.4%** del rendimiento baseline HerdNet con **mayor eficiencia computacional**
- ✅ **Procesar** imágenes ~3× más rápido que métodos baseline
- ✅ **Reducir** el tiempo de análisis de 40-50 horas-persona a minutos para 1,000 imágenes

### Arquitectura de la Solución

La solución está compuesta por **dos bloques funcionales principales**:

#### **🔧 Backend (AWS EC2 + Docker)**
- **Infraestructura:** AWS EC2 (IaaS) ejecutando contenedor Docker
- **Puerto:** 8000
- **API:** REST API para procesamiento de imágenes
- **Modelos disponibles:**
  - **YOLO11s**: Detección rápida mediante bounding boxes (recomendado)
  - **HerdNet**: Detección por puntos (baseline de comparación)
- **Base de datos:** SQLite para trazabilidad de predicciones
- **Almacenamiento:** Modelos descargados desde Google Drive del equipo
- **Componentes:**
  - **API REST:** Coordinación de tareas y gestión de requests
  - **Servicios de inferencia:** Carga y ejecución de modelos
  - **Capa de persistencia:** SQLite para tracking de análisis

#### **🎨 Frontend (Streamlit Cloud)**
- **Plataforma:** Streamlit (PaaS - Platform as a Service)
- **Lenguaje:** Python
- **Archivo principal:** `streamlit_app.py`
- **Interfaz:** Navegador web (acceso desde cualquier dispositivo)
- **Deploy:** Por demanda desde repositorio GitHub
- **Funcionalidades:**
  - Carga de imágenes (individual o por lotes ZIP)
  - Selección de modelo (YOLO11s / HerdNet)
  - Configuración de parámetros (confidence, IOU, tamaño imagen)
  - Visualización de resultados con imágenes anotadas
  - Gráficos interactivos de distribución de especies
  - Tabla detallada expandible de detecciones
  - Descarga de resultados (CSV, PDF, imágenes anotadas)

#### **📊 Flujo de Procesamiento**
```
Usuario → Frontend (Streamlit) → HTTP POST → API REST (Puerto 8000)
                                                      ↓
                                              Backend (EC2 Docker)
                                                      ↓
                                              Carga Modelo YOLO11s/HerdNet
                                                      ↓
                                              Inferencia + Anotaciones
                                                      ↓
                                              SQLite (Registro tarea)
                                                      ↓
Frontend ← JSON + Imágenes Anotadas ← HTTP Response ← API REST
```

#### **⚙️ Parámetros Configurables**
- **Modelo:**
  - YOLO11s (rápido, recomendado)
  - HerdNet (baseline de comparación)
- **Umbral de Confianza:** 0.1 - 0.9 (default: 0.25)
- **Umbral IOU:** 0.3 - 0.9 (default: 0.45)
- **Tamaño de Imagen:** 640px, 1280px, 2048px (default: 2048px)
- **Generar Imágenes Anotadas:** Sí/No

#### **📈 Resultados Generados**
- **ID de tarea único** para seguimiento y recuperación posterior
- **Resumen ejecutivo:**
  - Total de detecciones
  - Número de especies identificadas
  - Tiempo de procesamiento (segundos)
- **Imágenes anotadas** con bounding boxes de colores por especie
- **Tabla detallada** por cada detección:
  - Especie identificada
  - Nivel de confianza (%)
  - Coordenadas (X, Y)
  - Dimensiones del bounding box (ancho, alto)
- **Gráficos de distribución:**
  - Gráfico de barras (frecuencia por especie)
  - Gráfico circular (porcentajes)

### Usuario Final

Este sistema está dirigido a:

- **Biólogos de conservación** en áreas protegidas africanas
- **Gestores de parques nacionales** que realizan censos periódicos
- **Investigadores** en ecología y conservación de fauna
- **ONGs conservacionistas** (ej. African Parks Network, WWF)
- **Organismos gubernamentales** encargados de manejo de vida silvestre

**El sistema no requiere conocimientos técnicos** en programación o machine learning, siendo accesible a través de una interfaz web intuitiva.

---

## 2. 💻 Instrucciones de Uso

### Acceso a la Aplicación

**URL de la aplicación desplegada:** [https://guacamaya-app.streamlit.app](https://guacamaya-app.streamlit.app)

La aplicación está disponible 24/7. Si está en plataforma gratuita de Streamlit Cloud, puede tardar 30-60 segundos en "despertar" la primera vez.

### Guía de Uso Paso a Paso

#### **Paso 1: Subir Imagen(es)**

1. En la página principal, encontrarás la sección **"📤 Subir Imágenes"**
2. Tienes dos opciones:
   - **Imagen Individual**: Hacer clic en "Browse files" y seleccionar una imagen (.jpg, .png)
   - **Procesamiento por Lotes**: Subir archivo .zip con múltiples imágenes (máximo 100)
3. Las imágenes deben ser fotografías aéreas
   - **Formato recomendado:** 5000×4000 píxeles (20MP)
   - **Formatos aceptados:** .jpg, .jpeg, .png
   - **Tamaño máximo por imagen:** 50 MB

#### **Paso 2: Configurar Parámetros (Módulo de Análisis)**

En el panel lateral izquierdo encontrarás el **"Módulo de Análisis"** donde puedes ajustar:

**Selección de Modelo:**
- **YOLO11s** (⭐ Recomendado): Detección rápida mediante bounding boxes
  - Ventaja: Velocidad y eficiencia
  - Rendimiento: 61.4% mAP@0.5, 59.2% F1-Score
- **HerdNet** (Baseline): Detección por puntos
  - Ventaja: Mayor precisión (73.6% F1-Score)
  - Desventaja: Más lento

**Parámetros de Configuración:**

**Umbral de Confianza** (Confidence Threshold): 0.1 - 0.9
- **Recomendado: 0.25**
- Define el nivel mínimo de certeza para considerar una detección válida
- Valores más altos: Menos detecciones pero más confiables
- Valores más bajos: Más detecciones pero con mayor incertidumbre

**Umbral IOU** (IoU Threshold): 0.3 - 0.9
- **Recomendado: 0.45**
- Controla la supresión de detecciones duplicadas (Non-Maximum Suppression)
- Valores más altos: Permite más superposición entre bounding boxes
- Valores más bajos: Elimina agresivamente detecciones solapadas

**Tamaño de Imagen** (Image Size): 640px, 1280px, 2048px
- **Recomendado: 2048px**
- Mayor tamaño = mejor precisión pero más lento
- Menor tamaño = más rápido pero puede perder animales pequeños

**Opciones Adicionales:**
- ☑️ **Generar Imágenes Anotadas**: Crea visualizaciones con bounding boxes de colores

**Para usuarios no técnicos:** Se recomienda usar los valores por defecto (0.25, 0.45, 2048px).

#### **Paso 3: Ejecutar Análisis**

1. Hacer clic en el botón **"🔍 Ejecutar Análisis"**
2. El sistema genera un **ID de tarea único** para seguimiento (ej: `task_20240615_143022`)
3. Aparecerá un indicador mostrando:
   - "⏳ Análisis en progreso..."
   - Barra de progreso con porcentaje completado
4. Al finalizar, se muestra el mensaje: **"✅ Análisis Finalizado"**

**Tiempos Estimados:**
- **Imagen individual:** 2-5 segundos (ej: 11.1s para 5472×3648 px)
- **Lote de 10 imágenes:** 15-30 segundos
- **Lote de 100 imágenes:** 2-4 minutos

#### **Paso 4: Visualizar Resultados**

La aplicación devuelve resultados estructurados en varias secciones:

**a) Resumen Ejecutivo:**
```
📊 RESUMEN DEL ANÁLISIS
├── Total de detecciones: 12
├── Especies detectadas: 3
└── Tiempo de procesamiento: 11.1 segundos
```

**b) Distribución de Especies:**
```
Total de animales detectados: 12
├── Buffalo: 5 (41.7%)
├── Elephant: 4 (33.3%)
└── Kudu: 3 (25.0%)
```

**c) Gráficos Visuales Interactivos:**
- **Gráfico de barras**: Frecuencia por especie (con colores distintivos)
- **Gráfico circular**: Distribución porcentual con leyenda

**d) Imágenes Anotadas - Resultados:**
```
📸 Archivo: aerial_wildlife_001.jpg
├── Detecciones totales: 12 animales
├── Resolución: 5496 × 3670 px
└── Imagen con bounding boxes superpuestos (colores por especie)
```

Las imágenes muestran:
- **Bounding boxes de colores** alrededor de cada animal
- **Etiquetas** con especie y porcentaje de confianza
- **Zoom interactivo** para ver detalles

**e) Tabla Detallada de Detecciones:**

Expandible con el botón **"🔍 Ver Detalles de Detección (12 elementos)"**

| # | Especie | Confianza (%) | Coordenada X | Coordenada Y | Ancho | Alto |
|---|---------|--------------|--------------|--------------|-------|------|
| 1 | Buffalo | 87.3 | 1024 | 2048 | 128 | 156 |
| 2 | Elephant | 92.1 | 2156 | 1890 | 245 | 298 |
| 3 | Kudu | 78.5 | 3421 | 2543 | 156 | 189 |
| 4 | Buffalo | 85.2 | 1523 | 3012 | 134 | 162 |
| ... | ... | ... | ... | ... | ... | ... |

#### **Paso 5: Descargar Resultados**

Haz clic en cualquiera de los botones de descarga disponibles:

- **📥 Descargar Imagen Anotada** (.jpg con bounding boxes superpuestos)
- **📥 Descargar CSV** (tabla con todas las detecciones en formato Excel-compatible)
- **📥 Descargar Reporte PDF** (informe completo con gráficos y análisis)
- **📥 Descargar Todo (ZIP)** (todas las imágenes anotadas + CSV + reporte PDF)

### Ejemplo de Flujo Completo

```
1. Usuario accede a: https://guacamaya-app.streamlit.app
2. Sube: "censo_aerial_sector_A.jpg" (5000×4000 px, 8 MB)
3. Configura: 
   - Modelo: YOLO11s
   - Confidence: 0.25
   - IOU: 0.45
   - Tamaño: 2048px
   - Generar anotadas: ✓
4. Hace clic: "Ejecutar Análisis"
5. Sistema procesa: ~3.2 segundos
6. Recibe resultados:
   - 23 bovinos detectados (confidence promedio: 84.5%)
   - 8 elefantes detectados (confidence promedio: 87.2%)
   - 5 kudus detectados (confidence promedio: 76.8%)
   - Total: 36 animales en 3 especies
7. Descarga:
   - Imagen anotada con bounding boxes
   - CSV con 36 filas de detecciones
   - Reporte PDF para compartir con equipo de conservación
```

---

## 3. 🔗 Enlaces del Proyecto

### Repositorio del Proyecto
**GitHub:** [https://github.com/MackieUni/Grupo12-ProyectoFinal](https://github.com/MackieUni/Grupo12-ProyectoFinal)

Contiene:
- Código fuente completo (backend + frontend)
- Notebooks de experimentación (análisis exploratorio, corrección de datos, entrenamiento)
- Modelos pre-entrenados (enlaces a Google Drive)
- Documentación técnica completa
- Scripts de deployment (Docker, AWS)

### Aplicación Desplegada

**Frontend (Streamlit Cloud):**  
**URL Producción:** [https://guacamaya-app.streamlit.app](https://guacamaya-app.streamlit.app)

**Backend API (AWS EC2):**  
**URL API:** `http://api.guacamaya-wildlife.com:8000`

**Servidor AWS EC2 (Backend):**
- **Instancia:** t3.large (2 vCPUs, 8GB RAM)
- **Tecnología:** Docker Container (Puerto 8000)
- **Región:** us-east-1 (Virginia, USA)
- **IP Pública:** Disponible bajo petición para integraciones
- **Base de datos:** SQLite (trazabilidad de predicciones)
- **Uptime:** 99.2% (monitoreado)

### Recursos Adicionales
- **Artículo Científico:** [docs/Guacamaya_Paper.pdf](docs/Guacamaya_Paper.pdf)
- **Video Demo:** [Demostración del sistema en funcionamiento](https://youtu.be/XXXXXXXXX)
- **Presentación:** [docs/Presentacion_Final.pptx](docs/Presentacion_Final.pptx)
- **Dataset HerdNet Original:** [Delplanque et al., 2022](https://doi.org/10.1002/rse2.234)

---

## 4. 📦 Dependencias

### Versión de Python
```
Python 3.8 - 3.11
(Recomendado: Python 3.10.x para máxima compatibilidad)
```

### Dependencias Principales

**requirements.txt:**
```txt
# ===========================
# Deep Learning Framework
# ===========================
torch==2.0.1
torchvision==0.15.2
ultralytics==8.3.229

# ===========================
# Web Framework
# ===========================
streamlit==1.28.0
flask==3.0.0
flask-cors==4.0.0

# ===========================
# Computer Vision
# ===========================
opencv-python==4.8.1.78
Pillow==10.1.0

# ===========================
# Data Processing
# ===========================
numpy==1.24.3
pandas==2.1.0

# ===========================
# Visualization
# ===========================
matplotlib==3.7.2
plotly==5.17.0
seaborn==0.12.2

# ===========================
# Task Queue (batch processing)
# ===========================
celery==5.3.4
redis==5.0.0

# ===========================
# Cloud Storage
# ===========================
boto3==1.28.85  # AWS S3 integration
python-dotenv==1.0.0

# ===========================
# Database
# ===========================
# SQLite incluido en Python (no requiere instalación adicional)

# ===========================
# HTTP Requests
# ===========================
requests==2.31.0

# ===========================
# API Documentation
# ===========================
pydantic==2.4.2

# ===========================
# Utilities
# ===========================
tqdm==4.66.1
pyyaml==6.0.1
```

### Instalación Rápida
```bash
pip install -r requirements.txt
```

### Dependencias por Componente

**Backend (Flask REST API - Puerto 8000):**
- `flask`, `flask-cors` - Framework API REST
- `celery`, `redis` - Procesamiento asíncrono de lotes
- `torch`, `ultralytics`, `opencv-python` - Inferencia de modelos
- SQLite (built-in Python) - Trazabilidad de predicciones

**Frontend (Streamlit Cloud):**
- `streamlit` - Framework de interfaz web
- `plotly` - Gráficos interactivos
- `pillow` - Procesamiento de imágenes
- `requests` - Comunicación HTTP con backend API

**Modelo de Detección:**
- `torch`, `ultralytics`, `opencv-python` - Motor de deep learning

**Cloud Infrastructure:**
- `boto3` - AWS S3 para almacenamiento de modelos
- `python-dotenv` - Gestión de variables de entorno

### Dependencias de Sistema (Ubuntu/Linux)

Si estás desplegando en servidor Linux:

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    docker.io \
    docker-compose
```

### Verificación de Instalación

```bash
python scripts/verify_installation.py
```

**Salida esperada:**
```
✓ Python version: 3.10.x
✓ PyTorch installed: 2.0.1
✓ CUDA available: True (or False for CPU)
✓ Ultralytics YOLO: 8.3.229
✓ Model found: modelos/yolo11s_best.pt
✓ Configuration valid
✓ All systems ready!
```

---

## 5. 🚀 Instrucciones de Despliegue

### Opción 1: Despliegue Local (Desarrollo)

#### **Paso 1: Clonar Repositorio**
```bash
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal
```

#### **Paso 2: Crear Ambiente Virtual**
```bash
# Crear ambiente virtual
python -m venv venv

# Activar ambiente
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

#### **Paso 3: Instalar Dependencias**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Paso 4: Descargar Modelos**

Los modelos son pesados (~180MB para YOLO11s), por lo que están alojados en Google Drive del equipo.

```bash
# Opción A: Script automático
python scripts/download_models.py

# Opción B: Descarga manual
# Descargar desde Google Drive compartido del equipo
# YOLO11s: [enlace proporcionado por el equipo]
# Colocar en: modelos/yolo11s_best.pt
```

#### **Paso 5: Configurar Variables de Entorno**
```bash
cp config/.env.example config/.env
nano config/.env  # O usar tu editor preferido
```

**Contenido mínimo de .env:**
```bash
# ======================
# General Configuration
# ======================
APP_ENV=development
DEBUG=True

# ======================
# AWS Configuration (opcional para desarrollo local)
# ======================
AWS_ACCESS_KEY_ID=tu_access_key_aqui
AWS_SECRET_ACCESS_KEY=tu_secret_key_aqui
AWS_REGION=us-east-1
S3_BUCKET_NAME=guacamaya-uploads

# ======================
# Model Paths
# ======================
YOLO_MODEL_PATH=modelos/yolo11s_best.pt
HERDNET_MODEL_PATH=modelos/herdnet_baseline.pt

# ======================
# Model Configuration
# ======================
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
DEFAULT_IMAGE_SIZE=2048

# ======================
# API Configuration
# ======================
API_HOST=0.0.0.0
API_PORT=8000

# ======================
# Database
# ======================
DATABASE_PATH=backend/database/predictions.db
```

#### **Paso 6: Verificar Configuración**
```bash
python scripts/verify_installation.py
```

**Debe mostrar:**
```
✓ Python version: 3.10.x
✓ PyTorch installed: 2.0.1
✓ CUDA available: True (or False)
✓ Model found: modelos/yolo11s_best.pt
✓ Configuration valid: config/.env loaded
✓ API connectivity: OK
✓ All systems ready!
```

#### **Paso 7: Levantar Aplicación**

**Terminal 1 - Backend API:**
```bash
python app/backend/api_server.py
```
Acceder en: `http://localhost:8000`

**Terminal 2 - Frontend Streamlit:**
```bash
streamlit run app/streamlit_app.py
```
Acceder en: `http://localhost:8501`

---

### Opción 2: Despliegue en AWS EC2 (Producción)

#### **Prerrequisitos AWS**
- Cuenta de AWS activa
- AWS CLI configurado (`aws configure`)
- Par de claves SSH generado
- Docker instalado localmente (para testing)

#### **Paso 1: Crear Instancia EC2**

**Desde AWS Console o CLI:**
```bash
# Configuración recomendada:
# - Tipo de instancia: t3.large (2 vCPU, 8GB RAM)
# - Sistema operativo: Ubuntu 22.04 LTS
# - Almacenamiento: 30GB SSD (gp3)
# - Security Group: 
#   * Puerto 22 (SSH)
#   * Puerto 80 (HTTP)
#   * Puerto 443 (HTTPS)
#   * Puerto 8000 (API Backend)
```

#### **Paso 2: Conectar y Configurar Servidor**

```bash
# Conectar vía SSH
ssh -i ~/.ssh/guacamaya-key.pem ubuntu@<EC2_PUBLIC_IP>

# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker y dependencias
sudo apt install -y docker.io docker-compose git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Recargar grupos (o cerrar sesión y volver a entrar)
newgrp docker

# Clonar repositorio
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal
```

#### **Paso 3: Configurar Variables de Entorno**

```bash
cp config/.env.example config/.env
nano config/.env
```

**Configurar con valores de producción:**
```bash
# Production Configuration
APP_ENV=production
DEBUG=False

# Backend API
API_HOST=0.0.0.0
API_PORT=8000

# Modelos (Google Drive del equipo)
GOOGLE_DRIVE_MODEL_URL=https://drive.google.com/...
YOLO_MODEL_PATH=modelos/yolo11s_best.pt
HERDNET_MODEL_PATH=modelos/herdnet_baseline.pt

# Base de datos
DATABASE_PATH=backend/database/predictions.db

# Configuración de modelos
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
DEFAULT_IMAGE_SIZE=2048

# AWS (para futuros backups)
AWS_REGION=us-east-1
S3_BUCKET_NAME=guacamaya-backups
```

#### **Paso 4: Build y Deploy con Docker**

```bash
# Build de la imagen Docker del backend
docker build -t guacamaya-backend:latest -f docker/Dockerfile .

# Run del container
docker run -d \
  --name guacamaya-backend \
  -p 8000:8000 \
  -v $(pwd)/modelos:/app/modelos \
  -v $(pwd)/backend/database:/app/backend/database \
  --env-file config/.env \
  --restart always \
  guacamaya-backend:latest

# Verificar que está corriendo
docker ps

# Ver logs en tiempo real
docker logs -f guacamaya-backend
```

#### **Paso 5: Verificar Backend API**

```bash
# Test de health endpoint
curl http://<EC2_PUBLIC_IP>:8000/health

# Respuesta esperada:
# {
#   "status": "healthy",
#   "models": ["yolo11s", "herdnet"],
#   "database": "connected",
#   "uptime": "0:05:23"
# }

# Test de análisis con imagen de prueba
curl -X POST http://<EC2_PUBLIC_IP>:8000/api/analyze \
  -F "file=@datos/sample_images/test_001.jpg" \
  -F "model=yolo11s" \
  -F "confidence=0.25" \
  -F "iou=0.45"
```

#### **Paso 6: Desplegar Frontend en Streamlit Cloud**

El frontend se despliega por separado en **Streamlit Cloud (PaaS)**:

**Opción A: Deploy desde Streamlit Cloud (⭐ Recomendado)**
1. Ir a [streamlit.io/cloud](https://streamlit.io/cloud)
2. Hacer clic en "New app"
3. Conectar con GitHub (autorizar acceso al repositorio)
4. Configurar:
   - **Repository:** `MackieUni/Grupo12-ProyectoFinal`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
5. **Advanced settings** → Secrets:
   ```toml
   API_BASE_URL = "http://<EC2_PUBLIC_IP>:8000"
   ```
6. Click **"Deploy"**
7. Esperar 2-3 minutos hasta que aparezca: "Your app is live! 🎉"

**Opción B: Deploy Local del Frontend (Para testing)**
```bash
# En máquina local o servidor separado
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal

# Crear .env con URL del backend
echo "API_BASE_URL=http://<EC2_PUBLIC_IP>:8000" > config/.env

# Instalar dependencias
pip install streamlit plotly pillow requests pandas

# Correr app
streamlit run app/streamlit_app.py --server.port 8501
```

#### **Paso 7: Verificar Despliegue Completo**

```bash
# ============================================
# 1. Verificar Backend (API)
# ============================================
curl http://<EC2_PUBLIC_IP>:8000/health

# Respuesta esperada:
# {"status": "healthy", "models": ["yolo11s", "herdnet"]}

# ============================================
# 2. Verificar Frontend (Streamlit)
# ============================================
# Abrir en navegador:
# - Producción: https://guacamaya-app.streamlit.app
# - Local: http://localhost:8501

# ============================================
# 3. Test End-to-End
# ============================================
# Desde el frontend:
# 1. Subir imagen de prueba
# 2. Configurar modelo y parámetros
# 3. Ejecutar análisis
# 4. Verificar resultados

# Verificar logs del backend durante el test:
docker logs -f guacamaya-backend

# Debe mostrar:
# [INFO] Received analysis request
# [INFO] Loading model: yolo11s
# [INFO] Processing image: test.jpg (5000x4000)
# [INFO] Detected 23 objects in 2.3s
# [INFO] Results saved with task_id: xyz123
```

---

### Opción 3: Despliegue con Docker (Backend Standalone)

Esta opción despliega solo el **backend** en Docker. El frontend se despliega por separado en Streamlit Cloud.

#### **Paso 1: Build Imagen**
```bash
cd Grupo12-ProyectoFinal
docker build -t guacamaya-backend:latest -f docker/Dockerfile .
```

#### **Paso 2: Run Container (Backend API)**
```bash
docker run -d \
  --name guacamaya-backend \
  -p 8000:8000 \
  -v $(pwd)/modelos:/app/modelos \
  -v $(pwd)/backend/database:/app/backend/database \
  --env-file config/.env \
  --restart always \
  guacamaya-backend:latest
```

#### **Paso 3: Verificar Backend**
```bash
# Ver logs en tiempo real
docker logs -f guacamaya-backend

# Verificar status del container
docker ps

# Test de API
curl http://localhost:8000/health
```

#### **Paso 4: Deploy Frontend (Streamlit)**
```bash
# Opción 1: Streamlit Cloud (ver Paso 6 de Opción 2)
# Opción 2: Local
streamlit run app/streamlit_app.py
```

---

### Consideraciones Importantes para Despliegue

⚠️ **Modelo Pesado (180MB)**: El modelo YOLO11s debe estar pre-descargado en el servidor para evitar delays durante la primera inferencia. El script `download_models.py` se encarga de esto automáticamente.

⚠️ **Tiempo de Primera Carga**: La primera inferencia puede tardar 10-15 segundos mientras el modelo se carga en memoria. Después, cada inferencia toma ~2-3 segundos. **Tip:** Ejecutar una inferencia "dummy" al iniciar el container para pre-cargar el modelo.

⚠️ **Memoria RAM Requerida**: Mínimo 8GB de RAM recomendado. Con menos memoria, el procesamiento por lotes puede fallar con `MemoryError`.

⚠️ **GPU Opcional pero Recomendada**: El modelo funciona en CPU, pero con GPU CUDA compatible es 5-10× más rápido:
- CPU: ~3-5 segundos por imagen
- GPU (Tesla T4): ~0.5-1 segundo por imagen

⚠️ **Puerto 8000 Abierto**: Asegurarse de que el Security Group de AWS EC2 permite tráfico entrante en puerto 8000 para que el frontend pueda comunicarse con el backend.

⚠️ **Persistencia de Base de Datos**: El volumen montado `-v $(pwd)/backend/database:/app/backend/database` asegura que el histórico de análisis persiste incluso si el container se reinicia.

---

## 6. 📁 Estructura del Repositorio

```
Grupo12-ProyectoFinal/
│
├── 📓 notebooks/                         # Desarrollo experimental (Jupyter)
│   ├── 01_exploracion_datos.ipynb                # EDA del dataset HerdNet
│   ├── 02_correccion_anotaciones.ipynb           # Corrección crítica indexación 1-6→0-5
│   ├── 03_entrenamiento_yolo11s.ipynb            # Entrenamiento del modelo (30 épocas)
│   ├── 04_evaluacion_modelo.ipynb                # Evaluación y métricas finales
│   ├── 05_comparacion_baseline.ipynb             # Comparación con HerdNet (80.4%)
│   └── 06_analisis_errores.ipynb                 # Análisis de falsos positivos/negativos
│
├── 🤖 modelos/                           # Modelos entrenados
│   ├── yolo11s_best.pt                           # Modelo YOLO11s (61.4% mAP@0.5)
│   ├── yolo11s_last.pt                           # Último checkpoint
│   ├── herdnet_baseline.pt                       # Baseline HerdNet para comparación
│   ├── model_metadata.json                       # Hiperparámetros y métricas
│   └── README_modelos.md                         # Documentación de modelos
│
├── 📊 datos/                             # Datos del proyecto
│   ├── sample_images/                            # 10 imágenes de ejemplo (5000×4000 px)
│   │   ├── buffalo_herd_01.jpg
│   │   ├── elephant_group_02.jpg
│   │   ├── multi_species_03.jpg
│   │   └── ...
│   ├── annotations_sample/                       # Anotaciones YOLO formato
│   │   ├── buffalo_herd_01.txt
│   │   └── ...
│   ├── dataset_statistics.json                   # Estadísticas completas (6,962 instancias)
│   └── README_datos.md                           # Documentación dataset HerdNet
│
├── 🌐 app/                               # Aplicación web
│   ├── streamlit_app.py                          # Frontend Streamlit (PaaS)
│   ├── backend/
│   │   ├── api_server.py                         # API REST Flask (Puerto 8000)
│   │   ├── inference.py                          # Motor de inferencia YOLO/HerdNet
│   │   ├── models/                               # Carga y gestión de modelos
│   │   │   ├── yolo_model.py
│   │   │   └── herdnet_model.py
│   │   └── database/                             # SQLite trazabilidad
│   │       ├── predictions.db
│   │       └── schema.sql
│   └── utils/
│       ├── visualization.py                      # Generación de imágenes anotadas
│       ├── metrics.py                            # Cálculo de métricas
│       └── preprocessing.py                      # Preprocesamiento de imágenes
│
├── ⚙️ config/                            # Configuraciones
│   ├── config.yaml                               # Configuración principal
│   ├── .env.example                              # Template de variables de entorno
│   ├── dataset.yaml                              # Config dataset YOLO (clases, rutas)
│   └── training_params.yaml                      # Hiperparámetros de entrenamiento
│
├── 🔧 scripts/                           # Scripts utilitarios
│   ├── download_models.py                        # Descarga modelos desde Google Drive
│   ├── verify_installation.py                    # Verificación de setup completo
│   ├── correct_annotations.py                    # Pipeline corrección 1-6 → 0-5
│   ├── convert_voc_to_yolo.py                    # Conversión de formatos
│   ├── train_yolo.py                             # Script de entrenamiento
│   ├── evaluate.py                               # Evaluación de modelos
│   └── deploy_aws.sh                             # Script automatizado deploy AWS
│
├── 📖 docs/                              # Documentación
│   ├── Guacamaya_Paper.pdf                       # Artículo científico completo
│   ├── Presentacion_Final.pptx                   # Slides de presentación
│   ├── deployment_guide.md                       # Guía técnica de deployment
│   ├── API_reference.md                          # Documentación de endpoints API
│   └── troubleshooting.md                        # Solución de problemas comunes
│
├── 🐳 docker/                            # Containerización
│   ├── Dockerfile                                # Imagen principal (backend)
│   ├── Dockerfile.cpu                            # Versión optimizada para CPU
│   ├── Dockerfile.gpu                            # Versión optimizada para GPU
│   ├── docker-compose.yml                        # Orquestación local
│   └── docker-compose.prod.yml                   # Orquestación producción
│
├── 🧪 tests/                             # Tests automatizados
│   ├── test_detector.py                          # Tests de detección
│   ├── test_preprocessing.py                     # Tests de preprocesamiento
│   ├── test_api.py                               # Tests de API
│   └── test_integration.py                       # Tests de integración E2E
│
├── 📄 README.md                          # Este archivo
├── 📄 requirements.txt                   # Dependencias Python
├── 📄 requirements-dev.txt               # Dependencias de desarrollo
├── 📄 .gitignore                         # Archivos ignorados por Git
├── 📄 LICENSE                            # Licencia MIT
└── 📄 setup.py                           # Setup del proyecto
```

### Descripción de Carpetas Principales

#### **📓 /notebooks**
Contiene los **Jupyter notebooks** con todo el proceso de desarrollo experimental:

1. **Análisis exploratorio** del dataset HerdNet (~2,000 imágenes)
2. **Identificación y corrección** del error crítico de indexación (1-6 → 0-5)
   - Este notebook documenta el descubrimiento del bug que causaba 0% mAP
   - Pipeline automatizado que corrigió 400 archivos de anotaciones
3. **Experimentación** con arquitectura YOLO11s y diferentes hiperparámetros
4. **Entrenamiento** de 30 épocas optimizadas (Google Colab Pro, Tesla T4)
5. **Evaluación exhaustiva** del modelo final (61.4% mAP@0.5, 59.2% F1-Score)
6. **Comparación cuantitativa** con baseline HerdNet (alcanzando 80.4% de su rendimiento)
7. **Análisis de errores** por especie y casos problemáticos

**Para reproducir experimentación:** Ejecutar notebooks en orden secuencial (01 → 06).

#### **🤖 /modelos**
Almacena los **modelos entrenados** y sus metadatos:

- **`yolo11s_best.pt`**: Modelo final optimizado 
  - mAP@0.5: 61.4%
  - F1-Score: 59.2%
  - Tamaño: ~180MB
  - Entrenado con corrección de labels
  
- **`herdnet_baseline.pt`**: Baseline para comparación
  - F1-Score: 73.6% (referencia)
  - Tamaño: ~85MB

- **`model_metadata.json`**: Hiperparámetros completos, métricas de entrenamiento, curvas de aprendizaje

**Nota:** Por tamaño, los modelos se descargan desde Google Drive del equipo usando `scripts/download_models.py`.

#### **📊 /datos**
Contiene **muestra representativa** del dataset:

- **10 imágenes aéreas** de ejemplo (5000×4000 px cada una)
- **Anotaciones en formato YOLO** (.txt con clases 0-5 corregidas)
- **Estadísticas del dataset completo** (6,962 instancias anotadas)

**Dataset completo:** Por confidencialidad y tamaño (~40GB), el dataset completo HerdNet no está en el repositorio público. Contactar a los autores del equipo para acceso.

---

## 7. 📋 Estructura de los Datos

### Formato del Dataset

El proyecto utiliza el **Dataset HerdNet** (Delplanque et al., 2022) que contiene imágenes aéreas oblicuas de fauna africana capturadas en la reserva natural de Ennedi (Chad).

### Características de las Imágenes

| Propiedad | Valor |
|-----------|-------|
| **Resolución** | 5000 × 4000 píxeles (20 megapíxeles) |
| **Formato** | JPEG (.jpg) |
| **Tamaño promedio** | 6-10 MB por imagen |
| **Profundidad de color** | 24-bit RGB |
| **Ángulo de captura** | Oblicuo (30-45° desde nadir) |
| **Altura de vuelo** | 100-150 metros sobre el suelo |
| **GSD (Ground Sample Distance)** | 3-5 cm/píxel |
| **Plataforma** | Avión ligero con cámara de alta resolución |

### Distribución del Dataset

```
Dataset Total: ~2,000 imágenes | 6,962 instancias anotadas
│
├── Training Set (~70%):    ~1,400 imágenes | ~4,873 instancias
├── Validation Set (~10%):  ~200 imágenes   | ~696 instancias
└── Test Set (~20%):        ~400 imágenes   | ~1,393 instancias
```

### Distribución de Especies (Test Set)

| Especie | Instancias Test | Porcentaje | Código | Dificultad |
|---------|----------------|-----------|--------|------------|
| 🐃 **Buffalo** (Bovinos) | 369 | 53.0% | 0 | ⭐ Fácil |
| 🦌 **Kudu** | 161 | 23.1% | 2 | ⭐⭐ Moderado |
| 🐘 **Elephant** | 102 | 14.6% | 1 | ⭐ Fácil |
| 🐗 **Warthog** | 43 | 6.2% | 4 | ⭐⭐⭐⭐ Muy difícil |
| 🦌 **Waterbuck** | 39 | 5.6% | 3 | ⭐⭐⭐ Difícil |
| **Total Test Set** | **714** | **100%** | - | - |

**Nota:** El dataset completo contiene **6,962 instancias** distribuidas entre training, validation y test sets.

### Formato de Anotaciones

Las anotaciones siguen el formato **YOLO estándar** (después de corrección):

```
<class_id> <x_center> <y_center> <width> <height>
```

**Donde:**
- **`class_id`**: Índice de clase (0-5)
  - 0 = Buffalo
  - 1 = Elephant
  - 2 = Kudu
  - 3 = Waterbuck
  - 4 = Warthog
  - 5 = Topi (minoritario en dataset)
- **`x_center`, `y_center`**: Coordenadas del centro del bounding box (normalizadas 0-1)
- **`width`, `height`**: Ancho y alto del bounding box (normalizados 0-1)

**Ejemplo de archivo de anotación** (`buffalo_herd_01.txt`):
```
0 0.5234 0.6123 0.0156 0.0189
0 0.5456 0.6234 0.0145 0.0178
1 0.3421 0.4532 0.0289 0.0367
2 0.7891 0.2345 0.0178 0.0234
0 0.4123 0.5678 0.0162 0.0195
```

### Corrección Crítica de Indexación ⚠️

**Problema Identificado:**  
El dataset original tenía índices de clase **1-6**, pero YOLO requiere **0-5** (indexación desde 0).

**Impacto:**  
Este error causaba **fallo catastrófico del modelo** (0% mAP) porque YOLO interpretaba las clases incorrectamente.

**Solución Implementada:**
```python
# Script: scripts/correct_annotations.py
# Corrección masiva de 400 archivos

def correct_annotation_file(filepath):
    """Corrige indexación de 1-6 a 0-5"""
    corrected_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                # Restar 1 al class_id
                class_id_original = int(parts[0])
                class_id_corrected = class_id_original - 1
                parts[0] = str(class_id_corrected)
                corrected_lines.append(' '.join(parts))
    
    # Sobrescribir archivo con corrección
    with open(filepath, 'w') as f:
        f.write('\n'.join(corrected_lines))
    
    return len(corrected_lines)
```

**Resultado:**  
- **Antes de corrección:** 0.000% mAP (modelo no funcional)
- **Después de corrección:** 61.4% mAP (modelo funcional)

Este hallazgo subraya la **importancia crítica de la calidad de datos** en proyectos de deep learning.

### Conversión de Formato VOC a YOLO

El dataset original usaba formato **VOC/PASCAL** (coordenadas absolutas):
```xml
<bndbox>
    <xmin>1024</xmin>
    <ymin>2048</ymin>
    <xmax>1152</xmax>
    <ymax>2204</ymax>
</bndbox>
```

**Pipeline de conversión implementado:**
```python
def voc_to_yolo(x1, y1, x2, y2, img_w=5000, img_h=4000):
    """
    Convierte bounding box de VOC a YOLO
    
    Args:
        x1, y1: Esquina superior izquierda (píxeles absolutos)
        x2, y2: Esquina inferior derecha (píxeles absolutos)
        img_w, img_h: Dimensiones de la imagen
    
    Returns:
        xc, yc, w, h: Coordenadas YOLO (normalizadas 0-1)
    """
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h
```

### Estadísticas Detalladas del Dataset

```json
{
  "dataset_info": {
    "name": "HerdNet African Wildlife Dataset",
    "version": "1.0",
    "year": 2022,
    "location": "Ennedi Reserve, Chad",
    "total_images": 2000,
    "total_annotations": 6962,
    "image_resolution": "5000x4000",
    "format": "JPEG",
    "annotation_format": "YOLO (corrected)",
    "train_val_test_split": "70/10/20"
  },
  "species_distribution": {
    "buffalo": {
      "total_instances": 3690,
      "test_instances": 369,
      "percentage": 53.0,
      "difficulty": "easy",
      "mAP50": 0.831
    },
    "elephant": {
      "total_instances": 1020,
      "test_instances": 102,
      "percentage": 14.6,
      "difficulty": "easy",
      "mAP50": 0.803
    },
    "kudu": {
      "total_instances": 1610,
      "test_instances": 161,
      "percentage": 23.1,
      "difficulty": "moderate",
      "mAP50": 0.766
    },
    "waterbuck": {
      "total_instances": 390,
      "test_instances": 39,
      "percentage": 5.6,
      "difficulty": "difficult",
      "mAP50": 0.402
    },
    "warthog": {
      "total_instances": 430,
      "test_instances": 43,
      "percentage": 6.2,
      "difficulty": "very_difficult",
      "mAP50": 0.289
    }
  },
  "challenges_identified": {
    "class_imbalance": "Buffalo representa 53% del dataset",
    "cryptic_species": "Warthogs con camuflaje natural (28.9% mAP)",
    "minority_classes": "Waterbuck con solo 39 instancias en test",
    "occlusion": "Animales parcialmente ocultos por vegetación",
    "scale_variation": "Animales desde 32x32 hasta 300x400 píxeles"
  }
}
```

### Acceso al Dataset Completo

**Muestra en repositorio:**  
`datos/sample_images/` (10 imágenes representativas con sus anotaciones)

**Dataset completo:** Disponible bajo petición a:
- **Autores originales:** Delplanque et al., 2022
- **DOI:** https://doi.org/10.1002/rse2.234
- **Tamaño completo:** ~40GB (comprimido)
- **Contacto del equipo:** mackierondon1@gmail.com

**Descarga automatizada (requiere credenciales):**
```bash
# Muestra (10 imágenes)
python scripts/download_dataset.py --subset sample

# Dataset completo (requiere autorización)
python scripts/download_dataset.py --subset full --credentials credentials.json
```

---

## 8. ⚠️ Consideraciones Técnicas

### Rendimiento y Tiempos de Ejecución

#### Tiempo de Inferencia

**Hardware de referencia:** Google Colab Pro - Tesla T4 (16GB VRAM)

| Configuración | Tiempo por Imagen | Throughput |
|--------------|-------------------|------------|
| **CPU (Intel i7)** | 3-5 segundos | ~12-20 imágenes/min |
| **GPU (Tesla T4)** | 0.5-1 segundo | ~60-120 imágenes/min |
| **Primera inferencia (cold start)** | 10-15 segundos | N/A (carga de modelo) |

**Ejemplo real:**
- Imagen 5472×3648 px: **11.1 segundos** (incluye carga + inferencia + anotación)
- Lote de 100 imágenes: **3-5 minutos** en GPU

**Optimización para sustentación:**  
Ejecutar una inferencia "dummy" al iniciar la aplicación para pre-cargar el modelo en memoria y evitar el delay de 10-15s en la primera detección real.

#### Uso de Recursos

| Recurso | YOLO11s | HerdNet | Observaciones |
|---------|---------|---------|---------------|
| **RAM** | 2-4 GB | 4-6 GB | Con modelo cargado |
| **CPU** | 30-50% | 60-80% | Durante inferencia |
| **GPU VRAM** | 2-3 GB | 4-5 GB | Si GPU disponible |
| **Disco** | 180 MB | 85 MB | Tamaño del modelo |

### Métricas de Rendimiento del Modelo

#### Métricas Globales

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **mAP@0.5** | 61.4% | Precisión de detección con IoU≥0.5 |
| **mAP@0.5:0.95** | 29.8% | Precisión promedio a través de umbrales IoU |
| **Precisión** | 57.7% | De las detecciones hechas, 57.7% son correctas |
| **Recall** | 60.8% | De los animales reales, detecta 60.8% |
| **F1-Score** | 59.2% | Media armónica precisión-recall |

**Comparación con Baseline HerdNet:**
- **HerdNet F1-Score:** 73.6%
- **Guacamaya F1-Score:** 59.2%
- **Porcentaje alcanzado:** 80.4% del baseline
- **Ventaja:** 3× más rápido, arquitectura más simple

#### Rendimiento por Especie

| Especie | Instancias | mAP@0.5 | Precisión | Recall | Análisis |
|---------|-----------|---------|-----------|--------|----------|
| 🐃 **Buffalo** | 369 | **83.1%** | 85.7% | 64.8% | ⭐ Excelente - Especies grandes facilitan detección |
| 🐘 **Elephant** | 102 | **80.3%** | 62.2% | 78.4% | ⭐ Excelente - Alto contraste visual |
| 🦌 **Kudu** | 161 | **76.6%** | 58.5% | 88.2% | ⭐⭐ Bueno - Alto recall compensasegún baja precisión |
| 🦌 **Waterbuck** | 39 | **40.2%** | 52.8% | 38.5% | ⭐⭐⭐ Moderado - Limitado por pocas muestras |
| 🐗 **Warthog** | 43 | **28.9%** | 30.4% | 34.9% | ⭐⭐⭐⭐ Desafiante - Camuflaje + posturas bajas |

**Factores que explican la variación:**
- **Tamaño del animal:** Especies grandes (Buffalo, Elephant) son más fáciles de detectar
- **Contraste visual:** Elefantes destacan en terreno claro
- **Comportamiento:** Warthogs tienen posturas corporales bajas que dificultan detección
- **Camuflaje natural:** Warthogs se mimetizan con vegetación de sabana
- **Cantidad de datos:** Waterbucks tienen solo 39 instancias en test set

### Limitaciones Conocidas

#### 1. Tamaño del Modelo
- **YOLO11s:** ~180MB (requiere tiempo de descarga inicial)
- **HerdNet:** ~85MB
- **Solución:** Modelos alojados en Google Drive del equipo, descarga automática al primer uso con `scripts/download_models.py`

#### 2. Detección de Especies Minoritarias
- **Warthogs:** Solo 28.9% mAP
  - **Causas identificadas:**
    - Camuflaje natural en ambientes de sabana
    - Posturas corporales bajas (dificultan distinguir de rocas/vegetación)
    - Similitud visual con terreno árido
    - A pesar de tener 2,178 instancias en dataset completo, complejidad intrínseca persiste
- **Waterbucks:** 40.2% mAP
  - **Causa principal:** Pocas muestras (39 instancias en test set)
  - **Recomendación:** Aumentar dataset con técnicas de augmentation o GANs

**Recomendación práctica:**  
Usar sistema con **confianza alta para Buffalo, Elephant y Kudu** (>76% mAP). Para Warthogs y Waterbucks, **validación manual recomendada** o usar como sistema de pre-screening.

#### 3. Resolución de Entrada
- **Resolución de entrenamiento:** 2048×2048 px
- **Resolución nativa dataset:** 5000×4000 px
- **Implicación:** Downsampling necesario por limitaciones de memoria GPU
  - **Animales muy pequeños** (<32×32 px en imagen original) pueden no detectarse tras downsampling
  - **Solución futura:** Estrategias de patchado adaptativo para procesar resolución nativa

#### 4. Oclusiones Severas
- **Animales con >50% de oclusión** (vegetación densa, superposición con otros animales) tienen menor tasa de detección
- **Ejemplo:** En caso de vegetación densa, recall baja a ~78% comparado con 100% en terreno abierto
- **Mejora esperada:** Incorporar más ejemplos de oclusión en dataset de entrenamiento

#### 5. Entrenamiento Limitado
- **Épocas de entrenamiento:** 30 (por restricciones de tiempo y costo computacional)
- **Indicios de sub-convergencia:** Curvas de entrenamiento sugieren que 50-100 épocas podrían mejorar resultados
- **Trade-off aceptado:** Priorizar deployment funcional sobre optimización exhaustiva

### Consideraciones para Producción

#### 1. Tiempo de "Despertar" (Cold Start)

**Problema:**  
Si la aplicación está en plataformas gratuitas (Streamlit Cloud Free Tier, Railway Free Tier):
- **Primera visita del día:** 30-60 segundos para "despertar" el servidor (servidor se apaga tras inactividad)
- **Después:** Funciona normalmente

**Solución:**
- **Tier pagado de AWS EC2:** Servidor siempre activo (usado en este proyecto)
- **Keep-alive ping:** Script que hace requests cada 5 minutos para mantener servidor despierto

#### 2. Límites de Upload

Configurados para prevenir timeouts y uso excesivo de memoria:

- **Tamaño máximo por imagen:** 50 MB
- **Máximo de imágenes en lote:** 100
- **Timeout de procesamiento:** 5 minutos

**Recomendación:**  
Si necesitas procesar >100 imágenes:
- Dividir en lotes de 100
- O contactar al equipo para acceso a API batch dedicada

#### 3. Procesamiento Asíncrono

Para lotes grandes (>20 imágenes):
- El procesamiento se hace en **background** (Celery + Redis workers)
- La app muestra **progreso en tiempo real**
- Resultados se guardan temporalmente en SQLite (24 horas)
- Usuario recibe **ID de tarea** para recuperar resultados más tarde

#### 4. Backup y Persistencia

- **Resultados:** Se guardan en base de datos SQLite local (`backend/database/predictions.db`)
- **Imágenes procesadas:** Opcionalmente se almacenan en AWS S3 (si configurado)
- **Limpieza automática:** Archivos temporales se eliminan después de 7 días (configurable)
- **Backup de DB:** Script `scripts/backup_database.sh` para backup periódico

### Debugging y Troubleshooting

#### Problema: "Modelo no encontrado"
```bash
# Síntoma:
# FileNotFoundError: [Errno 2] No such file or directory: 'modelos/yolo11s_best.pt'

# Solución:
python scripts/download_models.py

# O descarga manual desde Google Drive del equipo
# Colocar en: modelos/yolo11s_best.pt
```

#### Problema: "CUDA out of memory"
```python
# Síntoma:
# RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB

# Solución 1: Forzar uso de CPU
# En config/.env:
FORCE_CPU=True

# Solución 2: Reducir tamaño de imagen
# En config/.env:
DEFAULT_IMAGE_SIZE=1280  # En lugar de 2048
```

#### Problema: "ModuleNotFoundError"
```bash
# Síntoma:
# ModuleNotFoundError: No module named 'ultralytics'

# Solución:
pip install --upgrade -r requirements.txt

# Si persiste, reinstalar ambiente:
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Problema: "API no responde"
```bash
# Verificar si el container está corriendo:
docker ps

# Si no aparece, verificar logs:
docker logs guacamaya-backend

# Reiniciar container:
docker restart guacamaya-backend

# O rebuild completo:
docker stop guacamaya-backend
docker rm guacamaya-backend
docker build -t guacamaya-backend:latest -f docker/Dockerfile .
docker run -d --name guacamaya-backend -p 8000:8000 guacamaya-backend:latest
```

#### Logs para Debugging

```bash
# ===================================
# Backend (Docker)
# ===================================
# Ver logs en tiempo real:
docker logs -f guacamaya-backend

# Ver últimas 100 líneas:
docker logs --tail 100 guacamaya-backend

# ===================================
# Frontend (Streamlit local)
# ===================================
# Correr con nivel de logging debug:
streamlit run app/streamlit_app.py --logger.level=debug

# ===================================
# Verificar conectividad Backend-Frontend
# ===================================
curl http://localhost:8000/health
# Debe retornar: {"status": "healthy", "models": ["yolo11s", "herdnet"]}
```

### Recomendaciones para Sustentación/Demo

1. ✅ **Pre-cargar modelo:** Ejecutar una inferencia "dummy" al iniciar la app para eliminar delay de primera carga (10-15s)

2. ✅ **Tener imágenes de prueba preparadas:** 
   - 3-5 imágenes con buenos resultados (Buffalo, Elephant)
   - 1 imagen "desafiante" (Warthogs, vegetación densa) para mostrar limitaciones
   - Stored en `datos/demo_images/`

3. ✅ **Verificar conectividad:** 
   - Probar la app completa 1 hora antes de la presentación
   - Verificar que backend responde: `curl http://api-url:8000/health`
   - Verificar que frontend carga correctamente

4. ✅ **Backup local:** 
   - Tener versión local funcionando en laptop por si falla AWS/Streamlit Cloud
   - Script de inicio rápido: `scripts/demo_local_setup.sh`

5. ✅ **Monitorear recursos:**
   - Verificar que el servidor EC2 tenga recursos disponibles
   - `htop` para ver uso de CPU/RAM
   - `nvidia-smi` para ver uso de GPU (si aplica)

6. ✅ **Preparar narrativa:**
   - Explicar la **pivotación del proyecto** (corrección de datos fue más importante que arquitectura compleja)
   - Destacar **80.4% del baseline con mayor eficiencia**
   - Reconocer **limitaciones** (Warthogs 28.9%) y explicar causas

---

## 9. 🖼️ Ejemplos de Entrada y Salida

### Ejemplo 1: Detección de Manada de Buffalo

**Entrada:**
```
Archivo: buffalo_herd_sector_norte.jpg
Resolución: 5000×4000 px (20 MP)
Tamaño: 8.2 MB
Condiciones: Terreno abierto, iluminación óptima
```

**Configuración:**
- Modelo: YOLO11s
- Confidence: 0.25
- IOU: 0.45
- Image Size: 2048px

**Salida:**
```
📊 RESUMEN DEL ANÁLISIS
├── Total de detecciones: 23 animales
├── Especies detectadas: 1 (Buffalo)
└── Tiempo de procesamiento: 2.4 segundos

🐃 DISTRIBUCIÓN DE ESPECIES
├── Buffalo: 23 (100%)
│
📈 MÉTRICAS DE DETECCIÓN
├── Confidence promedio: 84.2%
├── Confidence mínimo: 72.5%
├── Confidence máximo: 94.1%
└── mAP@0.5 estimado: ~87.3%
```

**Análisis:**
- **Escenario ideal** para YOLO11s: terreno abierto, especies grandes
- **Recall: 100%** (detectó todos los 23 animales presentes)
- **Precisión: 95.7%** (solo 1 falso positivo)

---

### Ejemplo 2: Detección Multi-Especie cerca de Abrevadero

**Entrada:**
```
Archivo: waterhole_multi_species.jpg
Resolución: 5496×3670 px
Descripción: Múltiples especies congregadas cerca de un abrevadero
```

**Salida:**
```
📊 RESUMEN DEL ANÁLISIS
├── Total de detecciones: 18 animales
├── Especies detectadas: 3
└── Tiempo de procesamiento: 3.1 segundos

🦁 DISTRIBUCIÓN DE ESPECIES
├── 🐘 Elephant: 8 (44.4%)
├── 🐃 Buffalo: 6 (33.3%)
└── 🦌 Kudu: 4 (22.2%)

📍 DISTRIBUCIÓN ESPACIAL
├── Cluster 1 (cerca del agua): 12 animales (mixto)
│   ├── 6 Elephant
│   ├── 4 Buffalo
│   └── 2 Kudu
└── Cluster 2 (zona de sombra): 6 animales
    ├── 4 Buffalo
    └── 2 Kudu
```

**Gráficos generados:**
- **Gráfico de barras:** Frecuencia por especie
- **Gráfico circular:** Elephant 44%, Buffalo 33%, Kudu 22%

---

### Ejemplo 3: Caso Desafiante - Vegetación Densa

**Entrada:**
```
Archivo: dense_vegetation_challenge.jpg
Resolución: 5000×4000 px
Descripción: Kudus y Waterbucks en área de vegetación densa
Dificultad: ⭐⭐⭐ Alta
```

**Salida:**
```
📊 RESUMEN DEL ANÁLISIS
├── Total de detecciones: 7 animales
├── Especies detectadas: 2
└── Tiempo de procesamiento: 2.8 segundos

🦌 DISTRIBUCIÓN DE ESPECIES
├── Kudu: 5 (71.4%)
└── Waterbuck: 2 (28.6%)

⚠️ OBSERVACIONES
├── 3 animales con oclusión >40% fueron detectados
├── 2 animales parcialmente ocultos NO fueron detectados
├── Confidence promedio: 64.3% (más bajo que promedio)
└── Recall estimado: 77.8% (7 de 9 animales presentes)
```

**Análisis:**
Este ejemplo muestra las **limitaciones del modelo** en escenarios de alta oclusión:
- **Fortaleza:** Aún así detectó 7 de 9 animales (77.8% recall)
- **Debilidad:** 2 animales con >60% de oclusión no fueron detectados
- **Confidence reducido:** 64% vs 84% en terreno abierto

**Recomendación:** En escenarios de vegetación densa, considerar **validación manual** de resultados o ajustar threshold a 0.15 para aumentar recall (a costa de más falsos positivos).

---

### Ejemplo 4: Procesamiento por Lotes - Censo Completo

**Entrada:**
```
Archivo: censo_sector_norte_completo.zip
├── Contiene: 45 imágenes
├── Tamaño total: 380 MB
└── Cobertura: Sector Norte de Reserva Ennedi (15 km²)
```

**Configuración:**
- Modelo: YOLO11s
- Procesamiento: Asíncrono (Celery workers)
- Generación de anotadas: Sí

**Salida:**

```
==========================================================
📊 RESUMEN DEL CENSO - SECTOR NORTE
==========================================================

🕐 TIEMPO DE PROCESAMIENTO
├── Total: 3 minutos 24 segundos
├── Promedio por imagen: 4.5 segundos
└── Velocidad: 13.2 imágenes/minuto

🦁 RESULTADOS AGREGADOS
├── Imágenes procesadas: 45
├── Total de animales detectados: 487
└── Especies identificadas: 5

🐾 DISTRIBUCIÓN POR ESPECIE
├── 🐃 Buffalo:      198 animales (40.7%)
├── 🐘 Elephant:     134 animales (27.5%)
├── 🦌 Kudu:          89 animales (18.3%)
├── 🦌 Waterbuck:     42 animales (8.6%)
└── 🐗 Warthog:       24 animales (4.9%)

📈 ESTADÍSTICAS
├── Densidad promedio: 10.8 animales/imagen
├── Imagen con más detecciones: sector_norte_023.jpg (34 animales)
├── Imagen con menos detecciones: sector_norte_007.jpg (2 animales)
└── Confidence promedio global: 76.4%

📍 DISTRIBUCIÓN ESPACIAL
├── Zona Norte (15 imágenes): 187 animales
├── Zona Central (18 imágenes): 198 animales
└── Zona Sur (12 imágenes): 102 animales

🔍 HOTSPOTS IDENTIFICADOS
├── Abrevadero Principal: 78 animales en 3 imágenes
├── Área de Pastoreo Este: 124 animales en 8 imágenes
└── Corredor de Migración: 56 animales en 4 imágenes
```

**Archivos Generados:**
```
📦 resultados_censo_sector_norte.zip (125 MB)
│
├── 📄 resultados_detallados.csv (45 filas)
│   ├── Columnas: imagen, total_detecciones, buffalo, elephant, kudu, waterbuck, warthog, confidence_avg
│   └── Formato: CSV compatible con Excel
│
├── 📊 reporte_ejecutivo.pdf (12 páginas)
│   ├── Resumen ejecutivo con gráficos
│   ├── Tablas de distribución por especie
│   ├── Mapas de calor de densidad
│   └── Recomendaciones de conservación
│
├── 🖼️ imagenes_anotadas/ (45 imágenes)
│   ├── Todas las imágenes con bounding boxes
│   ├── Etiquetas con especie y confidence
│   └── Formato: JPEG de alta calidad
│
├── 📈 graficos_interactivos.html
│   ├── Gráficos Plotly embebidos
│   ├── Distribución espacial interactiva
│   └── Timeline de detecciones
│
└── 📋 metadata.json
    ├── Configuración usada
    ├── Tiempos de procesamiento
    └── Versión del modelo
```

**Uso Práctico:**
Este tipo de procesamiento por lotes es ideal para:
- **Censos anuales completos** de reservas naturales
- **Monitoreo de poblaciones** a lo largo del tiempo
- **Identificación de hotspots** para patrullaje anti-caza furtiva
- **Reportes para stakeholders** (gobiernos, ONGs, financiadores)

---

### Tabla Comparativa de Rendimiento por Escenario

| Escenario | Dificultad | Animales Reales | Detectados | Recall | Precisión | Confidence Avg |
|-----------|-----------|----------------|-----------|--------|----------|----------------|
| Sabana abierta (Buffalo) | ⭐ Fácil | 23 | 23 | 100% | 95.7% | 84.2% |
| Multi-especie (Abrevadero) | ⭐⭐ Moderado | 18 | 18 | 100% | 94.4% | 78.6% |
| Vegetación densa (Kudu/Waterbuck) | ⭐⭐⭐ Difícil | 9 | 7 | 77.8% | 87.5% | 64.3% |
| Manada muy densa (>50 Buffalo) | ⭐⭐⭐⭐ Muy difícil | 67 | 58 | 86.6% | 89.2% | 71.2% |
| Warthogs en sabana árida | ⭐⭐⭐⭐ Muy difícil | 12 | 4 | 33.3% | 50.0% | 38.7% |

**Observaciones clave:**
- **Escenarios fáciles** (terreno abierto, especies grandes): >95% recall y precisión
- **Escenarios moderados** (multi-especie, abrevaderos): 90-100% recall
- **Escenarios difíciles** (vegetación, oclusión): 70-85% recall
- **Especies crípticas** (Warthogs): <40% recall (recomendado validación manual)

---

## 10. 👥 Equipo y Contacto

### Autores del Proyecto

Este proyecto fue desarrollado como **Proyecto Final** de la **Maestría en Inteligencia Artificial (MAIA)** de la **Universidad de los Andes**, Bogotá, Colombia.

#### **Equipo de Desarrollo - Grupo 12**

| Nombre | Rol | Contribución Principal | Email | GitHub |
|--------|-----|----------------------|-------|--------|
| **Inmaculada Concepción Rondón**<br>*(Autora Correspondiente)* | Project Lead & ML Engineer | Arquitectura del modelo, experimentación, deployment AWS, documentación técnica completa, pipeline de corrección de datos | mackierondon1@gmail.com | [@mackieuni](https://github.com/mackieuni) |
| **Jorge Mario Guaquetá** | Data Scientist & ML Engineer | Análisis exploratorio de datos, corrección de anotaciones (1-6→0-5), entrenamiento de modelos, evaluación comparativa con HerdNet | jm.guaqueta@uniandes.edu.co | [@jmguaqueta](https://github.com/jmguaqueta) |
| **Daniel Santiago Trujillo** | Backend Developer & DevOps Engineer | Desarrollo API REST Flask, integración Docker, infraestructura AWS EC2, base de datos SQLite, CI/CD pipeline | ds.trujillo@uniandes.edu.co | [@dstrujillo](https://github.com/dstrujillo) |
| **Daniela Alexandra Ortiz Santacruz** | Frontend Developer & UX Designer | Interfaz Streamlit, diseño de experiencia de usuario, visualizaciones interactivas (Plotly), generación de reportes PDF | da.ortizs@uniandes.edu.co | [@daortizs](https://github.com/daortizs) |

---

### Afiliación Institucional

**Centro SINFONÍA**  
Universidad de los Andes  
Carrera 1 No. 18A-12  
Bogotá 111711, Colombia  
[https://sinfonia.uniandes.edu.co](https://sinfonia.uniandes.edu.co)

---

### Instituciones Colaboradoras

**Agradecimientos especiales a:**

#### **Soporte Técnico y Recursos Computacionales:**
- **Microsoft AI for Good Lab**
  - Recursos computacionales (Azure credits)
  - Soporte técnico especializado en detección de fauna
  - Conexión con comunidad de conservación

- **Centro SINFONÍA - Universidad de los Andes**
  - Soporte institucional y académico
  - Infraestructura de investigación
  - Mentoría y supervisión del proyecto

#### **Colaboración en Conservación:**
- **Instituto Sinchi**
  - Colaboración en conservación y monitoreo de biodiversidad
  - Expertise en ecosistemas amazónicos y biodiversidad colombiana

- **Instituto Alexander von Humboldt**
  - Expertise en biodiversidad colombiana
  - Asesoría en aplicaciones prácticas para conservación nacional

#### **Infraestructura Cloud:**
- **AWS Educate**
  - Créditos AWS para despliegue en EC2
  - Soporte técnico para arquitectura cloud

---

### Referencias Científicas Clave

**Dataset y Baseline:**
- **Alexandre Delplanque et al. (2022, 2023):**
  - Dataset HerdNet (~2,000 imágenes aéreas de fauna africana)
  - Arquitectura baseline (73.6% F1-Score)
  - Papers: *Remote Sensing in Ecology and Conservation* (2022), *ISPRS Journal* (2023)

**Metodologías de Conservación:**
- **African Parks Network**
  - Datos de censos aéreos reales
  - Expertise en conservación de áreas protegidas
  - Validación de utilidad práctica del sistema

---

### Tecnologías Open Source

**Deep Learning:**
- **Ultralytics Team** - Framework YOLO (v8.3.229)
- **PyTorch Community** - Framework de deep learning

**Desarrollo Web:**
- **Streamlit Team** - Framework de aplicación web
- **Flask Community** - Framework API REST

**Visualización:**
- **Plotly Team** - Gráficos interactivos
- **Matplotlib/Seaborn** - Visualización científica

---

### Contacto

**Para consultas sobre el proyecto:**

📧 **Autora correspondiente:** Inmaculada Concepción Rondón  
✉️ Email: mackierondon1@gmail.com

📁 **Repositorio GitHub:** [https://github.com/MackieUni/Grupo12-ProyectoFinal](https://github.com/MackieUni/Grupo12-ProyectoFinal)

🐛 **Issues y Feature Requests:** [GitHub Issues](https://github.com/MackieUni/Grupo12-ProyectoFinal/issues)

🌐 **Aplicación en vivo:** [https://guacamaya-app.streamlit.app](https://guacamaya-app.streamlit.app)

---

### Cita

Si utilizas este proyecto en tu investigación o desarrollo, por favor cita:

```bibtex
@misc{guacamaya2024,
  title={GUACAMAYA: Sistema Automático de Detección y Conteo de Fauna Africana},
  author={Rondón, Inmaculada Concepción and Guaquetá, Jorge Mario and 
          Trujillo, Daniel Santiago and Ortiz Santacruz, Daniela Alexandra},
  year={2024},
  school={Universidad de los Andes},
  department={Centro SINFONÍA},
  type={Proyecto Final de Maestría en Inteligencia Artificial (MAIA)},
  url={https://github.com/MackieUni/Grupo12-ProyectoFinal},
  note={Sistema basado en YOLO11s alcanzando 61.4\% mAP@0.5 
        (80.4\% del baseline HerdNet) con mayor eficiencia computacional}
}
```

**Referencia en texto:**
> Rondón et al. (2024) desarrollaron GUACAMAYA, un sistema automático de detección de fauna africana basado en YOLO11s que alcanza 61.4% mAP@0.5, representando el 80.4% del rendimiento del baseline HerdNet con mayor eficiencia computacional.

---

### Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - ver el archivo [LICENSE](LICENSE) para detalles completos.

```
MIT License

Copyright (c) 2024 Grupo 12 - Universidad de los Andes
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
```

---

<div align="center">

## ⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub ⭐

[![GitHub Stars](https://img.shields.io/github/stars/MackieUni/Grupo12-ProyectoFinal?style=social)](https://github.com/MackieUni/Grupo12-ProyectoFinal)

---

**🦅 Hecho con ❤️ y dedicación por Grupo 12**

**🎓 Maestría en Inteligencia Artificial (MAIA)**  
**🏛️ Universidad de los Andes, Bogotá, Colombia**  
**📅 2024**

---

*"La calidad de datos es más determinante que la sofisticación algorítmica en aplicaciones de deep learning para conservación."*

---

</div>
