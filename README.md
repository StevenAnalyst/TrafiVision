# 🚦 TrafiVision — Monitoreo Inteligente de Tráfico

Sistema de detección y conteo automático de vehículos en intersecciones urbanas, aplicando visión por computadora a video o imágenes capturadas por cámaras de vigilancia ya existentes.

## 📌 Descripción del problema

**Caso ficticio:** la Secretaría de Movilidad de un municipio mediano quiere tomar decisiones de planeación vial (semáforos, horarios pico, necesidad de nuevos carriles) basadas en datos reales de flujo vehicular. Instalar sensores de conteo dedicados es costoso, pero el municipio ya cuenta con cámaras de seguridad en varias intersecciones. **TrafiVision** reutiliza ese video para detectar y contar automáticamente carros, motos, buses, camiones y bicicletas, generando estadísticas listas para análisis.

## 🏗️ Sector y contexto

- **Sector:** Transporte / movilidad urbana
- **Contexto:** caso ficticio pero verosímil de gestión de tráfico municipal con infraestructura de cámaras existente.

## 🧠 Modelo de visión por computadora utilizado

- **Modelo:** YOLOv8 (`yolov8n.pt` / `yolov8s.pt`), pre-entrenado sobre el dataset **COCO** mediante la librería [Ultralytics](https://github.com/ultralytics/ultralytics).
- **Justificación:** COCO ya incluye de forma nativa las clases `car`, `motorcycle`, `bus`, `truck`, `bicycle` y `person`, que cubren exactamente las necesidades del caso de uso. Esto permite usar el modelo **sin reentrenamiento**, cumpliendo el requisito de aplicar un modelo previamente entrenado. Además, YOLOv8 ofrece muy buen balance entre velocidad y precisión, lo que lo hace viable tanto para imágenes como para procesamiento de video casi en tiempo real.
- Para el conteo en video se usa el módulo de **tracking** integrado de Ultralytics (ByteTrack), que asigna un ID único a cada vehículo para evitar contarlo varias veces mientras permanece en cuadro.

## ⚙️ Funcionalidades

- **Modo Imagen:** sube una foto de una vía y obtén el conteo de vehículos por tipo, con bounding boxes y descarga de resultados en CSV.
- **Modo Video:** sube un clip de video y obtén:
  - Video anotado con detecciones y tracking.
  - Conteo total de vehículos únicos por tipo.
  - Gráfico de flujo de tráfico a lo largo del tiempo.
  - Exportación de resultados en CSV.
- Umbral de confianza ajustable y opción de incluir conteo de peatones.

## 💻 Instalación y ejecución local

### Requisitos previos
- Python 3.9 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/trafivision.git
cd trafivision

# 2. Crear un entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`. La primera vez que se ejecuta, Ultralytics descargará automáticamente los pesos del modelo YOLOv8 seleccionado.

## ☁️ Despliegue

La aplicación puede desplegarse gratuitamente en cualquiera de estas plataformas:

### Opción A: Streamlit Community Cloud (recomendado)
1. Sube el repositorio a GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Selecciona "New app", elige el repositorio, la rama y `app.py` como archivo principal.
4. Click en "Deploy". En unos minutos la app estará pública.

### Opción B: Hugging Face Spaces
1. Crea un nuevo Space con SDK "Streamlit".
2. Sube `app.py` y `requirements.txt`.
3. El Space se construye y despliega automáticamente.

En nuestro caso esta desplegado en Streamliit community Cloud

**🔗 Enlace a la aplicación desplegada:** _[completar con el enlace una vez desplegada]_

## 📁 Estructura del repositorio

```
trafivision/
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias del proyecto
└── README.md            # Este archivo
```

## 👥 Integrantes del equipo 

Anderson alonso cano arboleda
Brandon alexis quintero alvares
Mateo bejarano mejia
Steven tobon londoño
Valery restrepo alvarez

## 📊 Notas técnicas

- El conteo en imágenes es por detección directa (no acumulativo).
- El conteo en video usa tracking para contar **vehículos únicos**, evitando duplicados entre cuadros consecutivos.
- El parámetro "Procesar cada N cuadros" permite balancear velocidad de procesamiento contra precisión del conteo en videos largos.
