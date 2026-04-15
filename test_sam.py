import gradio as gr
import numpy as np
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import json
from datetime import datetime

try:
    # Importa Grounding DINO se disponibile
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util import box_ops
    import groundingdino.datasets.transforms as T

    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("⚠️ Grounding DINO non disponibile - funzionerà solo la modalità manuale")


# === CONFIGURAZIONE ===
class Config:
    # SAM Config
    SAM_MODEL_TYPE = "vit_h"
    SAM_CHECKPOINT = "/models/sam_vit_h_4b8939.pth"

    # Grounding DINO Config

    DINO_CONFIG_PATH = (
        ".venv/Lib/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"

    # Detection Config
    TEXT_PROMPT = (
        "pouch . doy pack . flexible packaging . food package . stand up pouch"
    )
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()


# === STATO GLOBALE ===
class AppState:
    def __init__(self):
        self.image = None
        self.input_points = []
        self.input_labels = []
        self.detected_boxes = []
        self.detection_scores = []
        self.current_mask = None
        self.processing_time = 0
        self.debug_info = {}

    def reset_points(self):
        self.input_points = []
        self.input_labels = []

    def reset_all(self):
        self.reset_points()
        self.detected_boxes = []
        self.detection_scores = []
        self.current_mask = None
        self.debug_info = {}


state = AppState()


# === CARICAMENTO MODELLI ===
def load_models():
    """Carica SAM e opzionalmente Grounding DINO"""
    try:
        # Carica SAM
        sam = sam_model_registry[config.SAM_MODEL_TYPE](
            checkpoint=config.SAM_CHECKPOINT
        ).to(config.DEVICE)
        predictor = SamPredictor(sam)

        # Carica Grounding DINO se disponibile
        dino_model = None
        if GROUNDING_DINO_AVAILABLE:
            try:
                cfg = SLConfig.fromfile(config.DINO_CONFIG_PATH)
                dino_model = build_model(cfg)
                checkpoint = torch.load(
                    config.DINO_CHECKPOINT, map_location=config.DEVICE
                )
                dino_model.load_state_dict(
                    clean_state_dict(checkpoint["model"]), strict=False
                )
                dino_model.eval().to(config.DEVICE)
            except Exception as e:
                print(f"⚠️ Errore caricamento Grounding DINO: {e}")
                dino_model = None

        return predictor, dino_model
    except Exception as e:
        raise Exception(f"Errore caricamento modelli: {e}")


# Inizializza modelli
try:
    predictor, dino_model = load_models()
    print("✅ Modelli caricati con successo")
except Exception as e:
    print(f"❌ Errore: {e}")
    predictor, dino_model = None, None


# === FUNZIONI GROUNDING DINO ===
def preprocess_image_dino(image_pil):
    """Preprocessa immagine per Grounding DINO"""
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image_pil, None)
    return image_transformed


def detect_pouches(image_pil):
    """Rileva buste usando Grounding DINO"""
    if not GROUNDING_DINO_AVAILABLE or dino_model is None:
        return [], [], 0

    start_time = time.time()

    try:
        # Preprocessa immagine
        image_tensor = preprocess_image_dino(image_pil).unsqueeze(0).to(config.DEVICE)

        # Esegui detection
        with torch.no_grad():
            outputs = dino_model(image_tensor, captions=[config.TEXT_PROMPT])

        # Estrai risultati
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
        prediction_boxes = outputs["pred_boxes"].cpu()[0]

        # Filtra per soglia
        mask = prediction_logits.max(dim=1)[0] > config.BOX_THRESHOLD
        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]

        # Converti boxes in formato xyxy
        H, W = image_pil.size[1], image_pil.size[0]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        processing_time = time.time() - start_time
        scores = logits.max(dim=1)[0].tolist()

        return boxes.tolist(), scores, processing_time

    except Exception as e:
        print(f"Errore detection: {e}")
        return [], [], 0


# === FUNZIONI VISUALIZZAZIONE ===
def create_debug_overlay(image, boxes, scores, points, labels):
    """Crea overlay di debug con informazioni dettagliate"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(
        f"Debug: {len(boxes)} buste rilevate, {len(points)} punti",
        fontsize=14,
        fontweight="bold",
    )

    # Disegna bounding boxes
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 10,
            f"Busta {i + 1}: {score:.2f}",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
            color="white",
            fontweight="bold",
        )

    # Disegna punti di click
    for i, (point, label) in enumerate(zip(points, labels)):
        color = "lime" if label == 1 else "red"
        marker = "o" if label == 1 else "x"
        ax.plot(
            point[0],
            point[1],
            marker=marker,
            color=color,
            markersize=15,
            markeredgewidth=3,
        )
        ax.text(
            point[0] + 10, point[1] + 10, f"P{i + 1}", color=color, fontweight="bold"
        )

    ax.axis("off")

    # Converti in immagine
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return buf


def create_info_panel():
    """Crea pannello informativo"""
    info = f"""
    ## 📊 Informazioni Sistema
    
    **Stato Modelli:**
    - SAM: {"✅ Caricato" if predictor else "❌ Errore"}
    - Grounding DINO: {"✅ Disponibile" if GROUNDING_DINO_AVAILABLE and dino_model else "❌ Non disponibile"}
    - Device: {config.DEVICE}
    
    **Statistiche Sessione:**
    - Punti inseriti: {len(state.input_points)}
    - Buste rilevate: {len(state.detected_boxes)}
    - Ultimo processing: {state.processing_time:.2f}s
    
    **Parametri Detection:**
    - Box Threshold: {config.BOX_THRESHOLD}
    - Text Threshold: {config.TEXT_THRESHOLD}
    - Prompt: "{config.TEXT_PROMPT}"
    """
    return info


# === FUNZIONI PRINCIPALI ===
def load_image(img):
    """Carica e processa una nuova immagine"""
    if img is None:
        return None, None, create_info_panel(), "❌ Nessuna immagine caricata"

    state.reset_all()
    state.image = np.array(img)

    # Configura SAM
    if predictor:
        predictor.set_image(state.image)

    # Auto-detection con Grounding DINO
    if GROUNDING_DINO_AVAILABLE and dino_model:
        boxes, scores, proc_time = detect_pouches(img)
        state.detected_boxes = boxes
        state.detection_scores = scores
        state.processing_time = proc_time

        status = (
            f"✅ Immagine caricata - {len(boxes)} buste rilevate in {proc_time:.2f}s"
        )
    else:
        status = (
            "✅ Immagine caricata - Modalità manuale (Grounding DINO non disponibile)"
        )

    debug_img = create_debug_overlay(
        state.image,
        state.detected_boxes,
        state.detection_scores,
        state.input_points,
        state.input_labels,
    )

    return state.image, debug_img, create_info_panel(), status


def click_point(evt: gr.SelectData, label_type: str):
    """Gestisce click dell'utente per aggiungere punti"""
    if state.image is None or not predictor:
        return None, None, create_info_panel(), "❌ Carica prima un'immagine"

    # Aggiungi punto
    state.input_points.append([evt.index[0], evt.index[1]])
    state.input_labels.append(1 if label_type == "positive" else 0)

    # Esegui segmentazione
    start_time = time.time()
    try:
        input_points_np = np.array(state.input_points)
        input_labels_np = np.array(state.input_labels)

        masks, scores, _ = predictor.predict(
            point_coords=input_points_np,
            point_labels=input_labels_np,
            multimask_output=True,
        )

        # Seleziona migliore maschera
        best_idx = np.argmax(scores)
        state.current_mask = masks[best_idx]

        # Crea overlay
        overlay = state.image.copy()
        overlay[state.current_mask] = [0, 255, 0]  # Verde per la maschera

        proc_time = time.time() - start_time
        state.processing_time = proc_time

        status = f"✅ Segmentazione completata in {proc_time:.3f}s (Score: {scores[best_idx]:.3f})"

    except Exception as e:
        overlay = state.image
        status = f"❌ Errore segmentazione: {e}"

    # Aggiorna debug
    debug_img = create_debug_overlay(
        state.image,
        state.detected_boxes,
        state.detection_scores,
        state.input_points,
        state.input_labels,
    )

    return overlay, debug_img, create_info_panel(), status


def auto_segment_detected():
    """Segmenta automaticamente le buste rilevate"""
    if not state.detected_boxes or not predictor:
        return state.image, None, create_info_panel(), "❌ Nessuna busta rilevata"

    try:
        # Usa il centro della prima bounding box come punto positivo
        box = state.detected_boxes[0]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        state.reset_points()
        state.input_points = [[center_x, center_y]]
        state.input_labels = [1]

        # Segmenta
        masks, scores, _ = predictor.predict(
            point_coords=np.array(state.input_points),
            point_labels=np.array(state.input_labels),
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        state.current_mask = masks[best_idx]

        overlay = state.image.copy()
        overlay[state.current_mask] = [0, 255, 0]

        debug_img = create_debug_overlay(
            state.image,
            state.detected_boxes,
            state.detection_scores,
            state.input_points,
            state.input_labels,
        )

        status = f"✅ Auto-segmentazione completata (Score: {scores[best_idx]:.3f})"

        return overlay, debug_img, create_info_panel(), status

    except Exception as e:
        return (
            state.image,
            None,
            create_info_panel(),
            f"❌ Errore auto-segmentazione: {e}",
        )


def reset_points():
    """Reset dei punti di click"""
    state.reset_points()
    state.current_mask = None

    debug_img = create_debug_overlay(
        state.image,
        state.detected_boxes,
        state.detection_scores,
        state.input_points,
        state.input_labels,
    )

    return state.image, debug_img, create_info_panel(), "🔄 Punti resettati"


def export_mask():
    """Esporta la maschera corrente"""
    if state.current_mask is None:
        return None, "❌ Nessuna maschera da esportare"

    try:
        # Crea maschera binaria
        mask_img = (state.current_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_img)

        # Salva temporaneamente
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"maschera_busta_{timestamp}.png"

        return mask_pil, f"✅ Maschera esportata: {filename}"

    except Exception as e:
        return None, f"❌ Errore esportazione: {e}"


def export_debug_info():
    """Esporta informazioni di debug"""
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "image_shape": state.image.shape if state.image is not None else None,
        "detected_boxes": state.detected_boxes,
        "detection_scores": state.detection_scores,
        "input_points": state.input_points,
        "input_labels": state.input_labels,
        "processing_time": state.processing_time,
        "config": {
            "box_threshold": config.BOX_THRESHOLD,
            "text_threshold": config.TEXT_THRESHOLD,
            "text_prompt": config.TEXT_PROMPT,
        },
    }

    return json.dumps(debug_data, indent=2)


# === INTERFACCIA GRADIO ===
def create_interface():
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="🎯 Advanced Pouch Segmentation") as demo:
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #10b981, #3b82f6); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🎯 Sistema Avanzato Segmentazione Buste</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Grounding DINO + SAM per rilevamento e segmentazione automatica</p>
        </div>
        """)

        # Status bar
        status_bar = gr.Textbox(
            label="📊 Status",
            value="Pronto per iniziare",
            interactive=False,
            container=True,
        )

        with gr.Row():
            # Colonna principale
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 📤 Upload Immagine")
                    img_input = gr.Image(
                        type="pil", label="Carica immagine busta", height=400
                    )

                with gr.Group():
                    gr.Markdown("### 🎛️ Controlli")
                    with gr.Row():
                        btn_auto = gr.Button(
                            "🤖 Auto-Segmenta", variant="primary", size="lg"
                        )
                        btn_reset = gr.Button("🔄 Reset Punti", variant="secondary")
                        btn_export = gr.Button("💾 Esporta Maschera", variant="stop")

            # Colonna risultati
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 🎨 Risultato Segmentazione")
                    result_image = gr.Image(
                        label="Clicca per aggiungere punti positivi/negativi",
                        height=400,
                    )

                with gr.Row():
                    btn_pos = gr.Button("➕ Punto Positivo", variant="primary")
                    btn_neg = gr.Button("➖ Punto Negativo", variant="secondary")

        # Sezione Debug
        with gr.Group():
            gr.Markdown("### 🔍 Debug & Monitoring")
            with gr.Row():
                debug_image = gr.Image(label="Vista Debug", height=300)
                info_panel = gr.Markdown(create_info_panel())

        # Output files
        with gr.Group():
            gr.Markdown("### 📁 Download")
            with gr.Row():
                mask_file = gr.File(label="💾 Maschera PNG", file_types=[".png"])
                debug_file = gr.Textbox(label="📊 Debug JSON", lines=10, max_lines=20)

        # Eventi
        img_input.upload(
            load_image,
            inputs=[img_input],
            outputs=[result_image, debug_image, info_panel, status_bar],
        )

        result_image.select(
            lambda evt: click_point(evt, "positive"),
            outputs=[result_image, debug_image, info_panel, status_bar],
        )

        btn_pos.click(
            lambda evt: click_point(evt, "positive")
            if evt
            else (None, None, create_info_panel(), "❌ Clicca sull'immagine"),
            outputs=[result_image, debug_image, info_panel, status_bar],
        )

        btn_neg.click(
            lambda evt: click_point(evt, "negative")
            if evt
            else (None, None, create_info_panel(), "❌ Clicca sull'immagine"),
            outputs=[result_image, debug_image, info_panel, status_bar],
        )

        btn_auto.click(
            auto_segment_detected,
            outputs=[result_image, debug_image, info_panel, status_bar],
        )

        btn_reset.click(
            reset_points, outputs=[result_image, debug_image, info_panel, status_bar]
        )

        btn_export.click(export_mask, outputs=[mask_file, status_bar])

        # Debug export (aggiornamento automatico)
        demo.load(lambda: export_debug_info(), outputs=[debug_file])

    return demo


# === AVVIO ===
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, debug=True, show_error=True, server_port=7860)
