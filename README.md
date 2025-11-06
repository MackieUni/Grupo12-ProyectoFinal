# Proyecto Final – Grupo 12
**Detección y Conteo con Despliegue de Solución en la Nube**

##  Objetivo del Proyecto
Evidenciar que la propuesta del grupo ha sido integrada en una **aplicación funcional y accesible**, disponible en una plataforma computacional (p. ej., Streamlit Cloud / Hugging Face / AWS), cumpliendo con los criterios de despliegue y documentación.

---

##  Estructura del Repositorio (mínima)
.
├── notebooks/ # Análisis, entrenamiento y pruebas
├── modelos/ # Artefactos del modelo final (.pt, .pkl, .h5)
├── datos/ # Muestras / estructura de datos (sin datos sensibles)
├── app/ # Aplicación (Streamlit/Gradio/Flask)
├── config/ # config.yaml y .env.example
├── docs/ # Guías (DEPLOYMENT.md) y material del informe
└── README.md



---

##  Tecnologías Principales
- Python 3.10+, PyTorch/Ultralytics, NumPy, Pandas
- Streamlit (UI), FastAPI (opcional), uvicorn
- YAML/`python-dotenv` para **parametrización**

---

##  Instalación y Ejecución Local

### 1 Clonar el repositorio
```bash
git clone https://github.com/MackieUni/Grupo12-ProyectoFinal.git
cd Grupo12-ProyectoFinal
2 Crear entorno e instalar dependencias ------

python -m venv .venv
source .venv/bin/activate     # En Windows: .venv\Scripts\activate
pip install -r requirements.txt  -------

3 Ejecutar la aplicación

streamlit run app/streamlit_app.py
 Asegúrate de configurar .env y config/config.yaml antes de ejecutar.

 Parametrización
Variables de entorno en .env (usar .env.example como plantilla).

Configuración en config/config.yaml (rutas, umbrales, tamaño de entrada, etc.).

Todos los parámetros deben reflejarse en la UI (sliders/selects) o en config.

 Despliegue en la Nube (resumen)
Opción A – Streamlit Cloud: conectar repo → “New app” → rama main → app/streamlit_app.py.

Opción B – Hugging Face Spaces (Gradio/Streamlit): crear Space → seleccionar framework → vincular repo.

Opción C – AWS/Render: Dockerfile + requirements.txt + startup cmd.

Ver docs/DEPLOYMENT.md para pasos detallados.

## Roles y Responsabilidades ## 
Integrante	Rol	Responsabilidades clave
## Inmaculada Concepción Rondón (Mackie)##	Coordinación / Despliegue	App Streamlit, guía de despliegue, README y control del repo
## Jorge Mario Guaqueta Restrepo ##	Modelo base & Métricas	Re-entrenar YOLOv8, registrar en W&B/MLflow, tablas y gráficos de métricas
## Daniel Santiago Trujillo ##	Her(d)Net & Post-proceso	Baseline con patching 1024, fusión de detecciones, comparativa con YOLO
## Daniela Alexandra Ortiz ##	Datos/EDA & Balanceo	Curación de splits, augmentations, negativos, conversión COCO→YOLO

 ## Checklist del Entregable (rubrica) ##
 Funcionamiento (35%): la app corre y cumple requerimientos.

 README completo (15%): descripción, dependencias, entorno, despliegue, credenciales de prueba.

 Parametrización (15%): .env / config.yaml documentados y usables.

 Usabilidad/Presentación (35%): interfaz clara, resultados interpretables, visualizaciones.

 Enlaces
Repositorio: https://github.com/MackieUni/Grupo12-ProyectoFinal

Aplicación desplegada: [agregar URL cuando esté activa]
  
Datos de origen / referencias: [enlaces a datasets públicos] 

