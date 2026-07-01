"""
TrafiVision - Sistema de Monitoreo Inteligente de Tráfico
Taller de Visión por Computadora

Detecta y cuenta vehículos (carros, motos, buses, camiones, bicicletas)
en imágenes, video o cámara en vivo, usando un modelo YOLOv8 pre-entrenado (COCO).
"""

import tempfile
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="TrafiVision | Centro de Monitoreo",
    page_icon="🚦",
    layout="wide",
)

VEHICLE_CLASSES = {
    1: "Bicicleta",
    2: "Carro",
    3: "Moto",
    5: "Bus",
    7: "Camión",
}
PEDESTRIAN_CLASS = {0: "Peatón"}

CLASS_COLORS_BGR = {
    "Bicicleta": (71, 99, 255),
    "Carro": (235, 162, 54),
    "Moto": (86, 206, 255),
    "Bus": (255, 102, 153),
    "Camión": (64, 159, 255),
    "Peatón": (192, 192, 75),
}
CLASS_COLORS_HEX = {
    "Bicicleta": "#FF6347",
    "Carro": "#36A2EB",
    "Moto": "#FFCE56",
    "Bus": "#9966FF",
    "Camión": "#FF9F40",
    "Peatón": "#4BC0C0",
}

# ----------------------------------------------------------------------
# Estilos: tema "centro de control de tráfico"
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .tv-header {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(135deg, #11161F 0%, #1B2433 100%);
        border: 1px solid #2A3445; border-radius: 14px;
        padding: 18px 26px; margin-bottom: 18px;
    }
    .tv-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700;
        color: #F2F4F8; margin: 0; letter-spacing: 0.5px;
    }
    .tv-subtitle { color: #8893A6; font-size: 13px; margin-top: 2px; }
    .tv-status {
        display: flex; align-items: center; gap: 8px;
        font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #38D996;
        background: rgba(56,217,150,0.1); border: 1px solid rgba(56,217,150,0.35);
        padding: 6px 14px; border-radius: 999px;
    }
    .tv-dot {
        height: 9px; width: 9px; border-radius: 50%; background: #38D996;
        box-shadow: 0 0 0 0 rgba(56,217,150,0.7);
        animation: tv-pulse 1.6s infinite;
    }
    @keyframes tv-pulse {
        0% { box-shadow: 0 0 0 0 rgba(56,217,150,0.6); }
        70% { box-shadow: 0 0 0 8px rgba(56,217,150,0); }
        100% { box-shadow: 0 0 0 0 rgba(56,217,150,0); }
    }
    div[data-testid="stMetric"] {
        background: #141923; border: 1px solid #232B38; border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace; color: #FFB020;
    }
    .tv-log {
        background: #0B0E13; border: 1px solid #232B38; border-radius: 10px;
        padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
        color: #9FE6B8; height: 180px; overflow-y: auto;
    }
    .tv-log-line { color: #C8CEDA; }
    .tv-log-time { color: #5AA9FF; margin-right: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelo YOLOv8 (solo la primera vez)...")
def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    return YOLO(model_name)


def get_active_classes(include_pedestrians: bool) -> dict:
    classes = dict(VEHICLE_CLASSES)
    if include_pedestrians:
        classes.update(PEDESTRIAN_CLASS)
    return classes


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def log_event(label, conf):
    st.session_state.eventos.insert(
        0, f"<span class='tv-log-time'>{datetime.now().strftime('%H:%M:%S')}</span>"
        f"<span class='tv-log-line'>{label} detectado · confianza {conf:.2f}</span>"
    )
    st.session_state.eventos = st.session_state.eventos[:60]


def registrar_deteccion(tipo, conteos, duracion_seg=None):
    """Guarda el resultado de UNA detección (imagen, video o tanda de cámara)
    como una entrada de historial, y la suma a los totales acumulados del turno."""
    conteos = {k: v for k, v in conteos.items() if v > 0}
    entrada = {
        "Tipo": tipo,
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Duración (s)": round(duracion_seg, 1) if duracion_seg is not None else "-",
        "Total": sum(conteos.values()),
        "Carro": conteos.get("Carro", 0),
        "Moto": conteos.get("Moto", 0),
        "Bus": conteos.get("Bus", 0),
        "Camión": conteos.get("Camión", 0),
        "Bicicleta": conteos.get("Bicicleta", 0),
        "Peatón": conteos.get("Peatón", 0),
    }
    st.session_state.historial.insert(0, entrada)
    for k, v in conteos.items():
        st.session_state.totales_sesion[k] += v
    st.session_state.analisis_realizados += 1
    if conteos:
        top_label = max(conteos, key=conteos.get)
        log_event(f"{tipo} registrada ({sum(conteos.values())} vehículos, predominante: {top_label})", 1.0)
    return entrada


# ----------------------------------------------------------------------
# Estado de sesión (estadísticas acumuladas del "turno de monitoreo")
# ----------------------------------------------------------------------
if "totales_sesion" not in st.session_state:
    st.session_state.totales_sesion = defaultdict(int)
if "analisis_realizados" not in st.session_state:
    st.session_state.analisis_realizados = 0
if "eventos" not in st.session_state:
    st.session_state.eventos = []
if "historial" not in st.session_state:
    st.session_state.historial = []
if "inicio_sesion" not in st.session_state:
    st.session_state.inicio_sesion = datetime.now()

# ----------------------------------------------------------------------
# Encabezado tipo centro de monitoreo
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="tv-header">
        <div>
            <p class="tv-title">🚦 TrafiVision</p>
            <p class="tv-subtitle">Centro de monitoreo de tráfico · Secretaría de Movilidad (caso simulado)</p>
        </div>
        <div class="tv-status"><span class="tv-dot"></span> SISTEMA ACTIVO · turno iniciado {st.session_state.inicio_sesion.strftime('%H:%M')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar - configuración
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
confianza = st.sidebar.slider("Umbral de confianza", 0.1, 0.9, 0.35, 0.05)
incluir_peatones = st.sidebar.checkbox("Incluir conteo de peatones", value=False)
modelo_size = st.sidebar.selectbox(
    "Tamaño del modelo YOLOv8",
    ["yolov8n.pt (rápido)", "yolov8s.pt (más preciso)"],
    index=0,
)
model_name = modelo_size.split(" ")[0]

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Acerca del proyecto"):
    st.markdown(
        "**Caso:** una alcaldía municipal reutiliza sus cámaras de "
        "seguridad existentes para medir el flujo vehicular, sin instalar "
        "sensores adicionales.\n\n"
        "**Modelo:** YOLOv8, pre-entrenado en COCO. Ya reconoce carros, "
        "buses, camiones, motos y bicicletas sin reentrenamiento.\n\n"
        "**Tracking:** ByteTrack evita contar dos veces el mismo vehículo "
        "en un video."
    )

if st.sidebar.button("🔄 Reiniciar estadísticas de sesión"):
    st.session_state.totales_sesion = defaultdict(int)
    st.session_state.analisis_realizados = 0
    st.session_state.eventos = []
    st.session_state.historial = []
    st.session_state.inicio_sesion = datetime.now()
    st.rerun()

active_classes = get_active_classes(incluir_peatones)
model = load_model(model_name)

# ----------------------------------------------------------------------
# Panel de estadísticas acumuladas de la sesión (vista "dashboard")
# Se calcula a partir de TODAS las detecciones hechas (historial completo:
# imágenes, videos y tandas de cámara en vivo).
# ----------------------------------------------------------------------
st.markdown("### 📊 Estadísticas del turno (todas las detecciones)")
k1, k2, k3, k4, k5 = st.columns(5)
total_general = sum(st.session_state.totales_sesion.values())
k1.metric("Detecciones registradas", st.session_state.analisis_realizados)
k2.metric("Vehículos detectados (total)", total_general)
k3.metric("Carros", st.session_state.totales_sesion.get("Carro", 0))
k4.metric("Motos", st.session_state.totales_sesion.get("Moto", 0))
k5.metric("Buses + Camiones", st.session_state.totales_sesion.get("Bus", 0) + st.session_state.totales_sesion.get("Camión", 0))

st.markdown("&nbsp;", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Pestañas: Imagen / Video / Cámara en vivo
# ----------------------------------------------------------------------
tab_img, tab_video, tab_camara = st.tabs(["📷 Imagen", "🎥 Video", "📸 Cámara en vivo"])

# ====================== MODO IMAGEN ======================
with tab_img:
    archivo = st.file_uploader("Sube una imagen de la vía o intersección", type=["jpg", "jpeg", "png"], key="img_upl")

    if archivo is not None:
        img = Image.open(archivo).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        with st.spinner("Detectando vehículos..."):
            results = model(frame, conf=confianza, verbose=False)[0]

        conteo = defaultdict(int)
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in active_classes:
                continue
            label_name = active_classes[cls_id]
            conf_val = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            color = CLASS_COLORS_BGR.get(label_name, (0, 255, 0))
            draw_box(annotated, xyxy, f"{label_name} {conf_val:.2f}", color)
            conteo[label_name] += 1
            log_event(label_name, conf_val)

        if conteo:
            registrar_deteccion("Imagen", conteo)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detecciones", use_container_width=True)
        with col2:
            st.markdown("#### Conteo de esta imagen")
            if conteo:
                df = pd.DataFrame({"Tipo": list(conteo.keys()), "Cantidad": list(conteo.values())}).sort_values("Cantidad", ascending=False)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.bar_chart(df.set_index("Tipo"))
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar conteo (CSV)", csv, "conteo_vehiculos.csv", "text/csv", key="csv_img")
            else:
                st.info("No se detectaron vehículos con el umbral de confianza actual.")
    else:
        st.info("Sube una imagen para comenzar el análisis.")

# ====================== MODO VIDEO ======================
with tab_video:
    archivo_v = st.file_uploader("Sube un video de la vía o intersección", type=["mp4", "mov", "avi"], key="vid_upl")
    frame_skip = st.slider("Procesar cada N cuadros (más rápido = más alto)", 1, 10, 3)

    if archivo_v is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(archivo_v.read())
        video_path = tfile.name

        if st.button("▶️ Procesar video"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps / frame_skip, (width, height))

            seen_ids = defaultdict(set)
            timeline = []
            progress = st.progress(0)
            status = st.empty()
            frame_idx = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    results = model.track(frame, conf=confianza, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
                    frame_counts = defaultdict(int)
                    if results.boxes is not None and results.boxes.id is not None:
                        for box, track_id in zip(results.boxes, results.boxes.id):
                            cls_id = int(box.cls[0])
                            if cls_id not in active_classes:
                                continue
                            label_name = active_classes[cls_id]
                            seen_ids[label_name].add(int(track_id))
                            frame_counts[label_name] += 1
                            color = CLASS_COLORS_BGR.get(label_name, (0, 255, 0))
                            draw_box(frame, box.xyxy[0].tolist(), f"{label_name} #{int(track_id)}", color)
                    writer.write(frame)
                    timeline.append({"cuadro": frame_idx, **frame_counts})

                frame_idx += 1
                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0))
                status.text(f"Procesando cuadro {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            elapsed = time.time() - start_time
            st.success(f"Procesamiento completo en {elapsed:.1f} segundos.")

            totales = {k: len(v) for k, v in seen_ids.items()}
            if totales:
                registrar_deteccion("Video", totales, duracion_seg=elapsed)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 🎥 Video anotado")
                st.video(out_path)
            with col2:
                st.markdown("#### Conteo total (vehículos únicos)")
                if totales:
                    df_tot = pd.DataFrame({"Tipo": list(totales.keys()), "Cantidad": list(totales.values())}).sort_values("Cantidad", ascending=False)
                    st.dataframe(df_tot, hide_index=True, use_container_width=True)
                    st.bar_chart(df_tot.set_index("Tipo"))
                    csv = df_tot.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Descargar conteo (CSV)", csv, "conteo_video.csv", "text/csv", key="csv_vid")
                else:
                    st.info("No se detectaron vehículos en el video.")

            if timeline:
                st.markdown("#### 📈 Flujo de tráfico en el tiempo")
                df_timeline = pd.DataFrame(timeline).fillna(0).set_index("cuadro")
                st.line_chart(df_timeline)
    else:
        st.info("Sube un video para comenzar el análisis.")

# ====================== MODO CÁMARA EN VIVO ======================
with tab_camara:
    st.markdown(
        "Transmite y analiza **video continuo real** desde la cámara del computador, igual que lo haría "
        "una cámara de vigilancia de tráfico. Define cuánto debe durar la detección (por ejemplo, 5 minutos) "
        "y al finalizar obtendrás un resultado consolidado, igual que con imagen o video."
    )
    st.caption(
        "⚠️ Este modo accede directamente a la cámara del equipo donde corre la app (vía OpenCV), por lo que "
        "funciona al ejecutar el proyecto localmente con `streamlit run app.py`. Si despliegan la app en la nube "
        "para que otros la prueben, usen las pestañas de Imagen o Video — la nube no tiene acceso a la cámara "
        "de quien visita la página."
    )

    if "camara_corriendo" not in st.session_state:
        st.session_state.camara_corriendo = False

    col_cfg1, col_cfg2 = st.columns(2)
    cam_index = col_cfg1.number_input("Índice de cámara (0 = principal)", min_value=0, max_value=5, value=0, step=1)
    duracion_min = col_cfg2.number_input("Duración de la detección (minutos)", min_value=1, max_value=30, value=5, step=1)

    col_btn1, col_btn2 = st.columns(2)
    iniciar = col_btn1.button("▶️ Iniciar detección en vivo", disabled=st.session_state.camara_corriendo, use_container_width=True)
    detener = col_btn2.button("⏹️ Detener y guardar resultado", disabled=not st.session_state.camara_corriendo, use_container_width=True)

    if iniciar:
        st.session_state.camara_corriendo = True
        st.session_state.camara_inicio = time.time()
        st.session_state.camara_duracion_objetivo = duracion_min * 60
        st.session_state.camara_conteo_acumulado = defaultdict(int)
        st.session_state.camara_frames = 0
        st.rerun()

    progreso = st.progress(0.0)
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    def finalizar_deteccion_camara():
        elapsed_real = time.time() - st.session_state.camara_inicio
        entrada = registrar_deteccion(
            "Cámara en vivo",
            st.session_state.camara_conteo_acumulado,
            duracion_seg=elapsed_real,
        )
        st.session_state.camara_corriendo = False
        st.session_state.ultima_deteccion_camara = entrada

    if detener and st.session_state.camara_corriendo:
        finalizar_deteccion_camara()
        st.rerun()

    if st.session_state.camara_corriendo:
        elapsed_total = time.time() - st.session_state.camara_inicio
        objetivo = st.session_state.camara_duracion_objetivo
        restante = max(objetivo - elapsed_total, 0)
        progreso.progress(min(elapsed_total / objetivo, 1.0))
        stats_placeholder.markdown(
            f"**⏱️ {int(elapsed_total)}s / {int(objetivo)}s** · vehículos acumulados en esta detección: "
            f"**{sum(st.session_state.camara_conteo_acumulado.values())}**"
        )

        if restante <= 0:
            finalizar_deteccion_camara()
            st.rerun()
        else:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(
                    "No se pudo acceder a la cámara. Verifica que ningún otro programa la esté usando "
                    "y que el sistema haya dado permiso de cámara a Python."
                )
                st.session_state.camara_corriendo = False
            else:
                chunk = min(3, restante)
                start_chunk = time.time()
                while time.time() - start_chunk < chunk:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("La cámara dejó de enviar cuadros.")
                        break

                    results = model(frame, conf=confianza, verbose=False)[0]
                    annotated = frame.copy()

                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in active_classes:
                            continue
                        label_name = active_classes[cls_id]
                        conf_val = float(box.conf[0])
                        color = CLASS_COLORS_BGR.get(label_name, (0, 255, 0))
                        draw_box(annotated, box.xyxy[0].tolist(), f"{label_name} {conf_val:.2f}", color)
                        st.session_state.camara_conteo_acumulado[label_name] += 1

                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Transmisión en vivo",
                        use_container_width=True,
                    )
                    st.session_state.camara_frames += 1
                cap.release()
                st.rerun()
    else:
        frame_placeholder.info("Configura la duración y pulsa 'Iniciar detección en vivo' para comenzar.")

        if st.session_state.get("ultima_deteccion_camara"):
            ult = st.session_state.ultima_deteccion_camara
            st.markdown("#### ✅ Resultado de la última detección en vivo")
            res_cols = st.columns(6)
            for col, label in zip(res_cols, ["Carro", "Moto", "Bus", "Camión", "Bicicleta", "Peatón"]):
                col.metric(label, ult[label])
            st.caption(f"Duración: {ult['Duración (s)']}s · Total: {ult['Total']} vehículos · Hora: {ult['Hora']}")

# ----------------------------------------------------------------------
# Historial de detecciones (todas las corridas: imagen, video, cámara)
# ----------------------------------------------------------------------
st.markdown("### 📜 Historial de detecciones")
if st.session_state.historial:
    df_hist = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_hist, hide_index=True, use_container_width=True)
    csv_hist = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar historial completo (CSV)", csv_hist, "historial_trafivision.csv", "text/csv", key="csv_hist")
else:
    st.info("Aún no se ha registrado ninguna detección. Los resultados de cada imagen, video o tanda de cámara en vivo aparecerán aquí.")

# ----------------------------------------------------------------------
# Registro de eventos (consola en vivo)
# ----------------------------------------------------------------------
st.markdown("### 🖥️ Registro de detecciones")
log_html = "<br>".join(st.session_state.eventos) if st.session_state.eventos else "<span class='tv-log-line'>Esperando detecciones...</span>"
st.markdown(f"<div class='tv-log'>{log_html}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Proyecto final · Taller de Visión por Computadora · TrafiVision")