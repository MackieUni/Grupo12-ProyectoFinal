# Proyecto Final – Grupo 12  
### Detección y Conteo con Despliegue de Solución en la Nube

---

## Objetivo del Proyecto
Evidenciar que la propuesta de nuestro grupo ha sido integrada en una **aplicación funcional y accesible**, disponible en una plataforma computacional (por ejemplo, Streamlit o AWS EC2), cumpliendo con los criterios de despliegue y documentación establecidos por la profesora **Haydemar María Núñez Castro**.

---

##  Estructura del Repositorio
.
├── notebooks/ # Cuadernos con análisis, entrenamiento y pruebas
├── modelos/ # Archivos del modelo final (.pt, .pkl, .h5)
├── datos/ # Muestras o estructura de los datos usados
├── config/ # Archivos de configuración y variables de entorno
├── app/ # Aplicación (Streamlit, Gradio o Flask)
├── docs/ # Guías y documentación adicional
└── README.md

yaml
Copy code

---

##  Integrantes del Grupo 12
| Nombre | Rol | Responsabilidades |
|---------|-----|--------------------|
|**Inmaculada Concepción Rondón (Mackie)** | Líder de Despliegue y Documentación | Crear la app Streamlit (FastAPI opcional) para inferencia y visualizaciones (bounding boxes / puntos / resumen de métricas). Elaborar README maestro, configuración por `.env` y `config.yaml`, guía de despliegue (Hugging Face / AWS / Render) y documento Word/PDF final con enlaces y autores. |
|**Jorge Mario Guaqueta Restrepo**  Líder de Modelo Base y Métricas | Reentrenar YOLOv8-m/n con más épocas e input 1024 + augmentations; registrar todo en MLflow/W&B. Implementar métricas de conteo (MAE, F1 por clase, Count Accuracy) y tabla comparativa semanal. Entregables: runs reproducibles, tabla “YOLOv8 vs semana previa” y gráficos. |
|**Daniel Santiago Trujillo** | Líder Her(d)Net y Post-Proceso | Montar baseline Her(d)Net con patching 1024×1024 + overlap 25% y clustering/NMS adaptado para fusionar detecciones. Comparar directamente con YOLO en el mismo split. Entregables: notebook “patch→full image”, script de consolidación y métricas. |
|**Daniela Alexandra Ortiz**. | Líder de Datos/EDA y Balanceo | Curar muestra train/val/test estratificada; realizar data augmentation fotométrica y geométrica; incluir negativos (sin fauna) para reducir falsos positivos. Implementar samplers balanceados por clase y reportar distribución actualizada. Entregables: carpeta `data/` lista, gráficos de distribución y script de conversión COCO→YOLO. |

> ***Todos:** cada integrante entrena 1 – 2 variantes de modelo, sube artefactos a `modelos/` y realiza *pull requests* siguiendo la plantilla del repositorio..*

---

## Tecnologías Principales
- **Lenguaje:** Python 3.10+  
- **Framework:** Streamlit / Gradio  
- **Librerías:** PyTorch, NumPy, Pandas, scikit-learn, OpenCV, YAML  
- **Infraestructura:** AWS / HuggingFace Spaces / Streamlit Cloud 
- **Control de versiones:** Git + GitHub Actions  

---

##  Instalación y Ejecución Local

###  Clonar el repositorio
```bash
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal

## Instalación y Ejecución Local

### 1️Clonar el repositorio
```bash
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal ------

2️ Crear entorno virtual e instalar dependencias
bash

python -m venv .venv
source .venv/bin/activate     # En Windows: .venv\Scripts\activate
pip install -r requirements.txt ------

3️ Ejecutar la aplicación
bash

streamlit run app/streamlit_app.py
 Nota: Asegúrate de tener configurado el archivo .env y el config.yaml antes de ejecutar la aplicación.




---



