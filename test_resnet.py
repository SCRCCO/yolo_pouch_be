import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import threading
import cv2
import time
import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T

# ---------------------------------------
# 1) Utility: Embedding backbone (ResNet50 pretrained)
# ---------------------------------------
backbone = models.resnet50(pretrained=True)
# Remove final classification head, so output is 2048-dimensional feature vector
backbone.fc = torch.nn.Identity()
backbone.eval()
# Preprocessing transform for backbone
transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_embedding(image: np.ndarray) -> np.ndarray:
    """
    Compute 2048-d embedding for a BGR image crop.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0)  # shape [1,3,224,224]
    with torch.no_grad():
        emb = backbone(tensor)
    return emb.squeeze(0).cpu().numpy()  # shape [2048]


# ---------------------------------------
# 2) Globals for prototypes
# ---------------------------------------
prototypes = {
    "front": None,  # numpy arrays
    "back": None,
}

# ---------------------------------------
# 3) tkinter GUI & YOLO segmentation setup
# ---------------------------------------
root = tk.Tk()
root.title("YOLO Front/Back Capture & Live Detection")

# Debug text area
debug_text = tk.Text(root, height=8, width=50)
debug_text.pack(pady=5)


def log(msg: str):
    debug_text.insert(tk.END, msg + "\n")
    debug_text.see(tk.END)


# YOLO segmentation model
seg_model = None

# ---------------------------------------
# 4) Capture front/back images
# ---------------------------------------
cap = None
front_img = None
back_img = None


def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        log("[DEBUG] Webcam inizializzata.")


def capture_side(side: str):
    """
    Capture one frame from webcam and store as front/back.
    """
    global front_img, back_img
    ret, frame = cap.read()
    if not ret:
        log("[ERROR] Impossibile leggere dalla webcam.")
        return
    # Salva immagine catturata per debug
    debug_path = f"debug_{side}_capture.jpg"
    cv2.imwrite(debug_path, frame)
    log(f"[DEBUG] Immagine {side} catturata e salvata in {debug_path}")
    if side == "front":
        front_img = frame.copy()
        log("[INFO] Foto FRONTE acquisita.")
        btn_front.config(state=tk.DISABLED)
        btn_back.config(state=tk.NORMAL)
    else:
        back_img = frame.copy()
        log("[INFO] Foto RETRO acquisita.")
        btn_back.config(state=tk.DISABLED)
        # Once both captured, compute prototypes
        compute_prototypes()


# ---------------------------------------
# 5) Compute prototypes (mean of embeddings)
# ---------------------------------------
def compute_prototypes():
    """
    Build prototype embeddings for front and back images.
    """
    global prototypes
    if front_img is None or back_img is None:
        log("[WARN] Serve sia front che back per il prototipo.")
        return
    emb_f = get_embedding(front_img)
    emb_b = get_embedding(back_img)
    prototypes["front"] = emb_f
    prototypes["back"] = emb_b
    log("[INFO] Prototipi calcolati. Sistema pronto all'inferenza live.")
    # Salva embedding per debug
    np.save("debug_front_embedding.npy", emb_f)
    np.save("debug_back_embedding.npy", emb_b)
    log("[DEBUG] Embedding fronte e retro salvati su file.")
    # Enable live detection
    btn_live.config(state=tk.NORMAL)


# ---------------------------------------
# 6) Live detection loop
def live_detection_loop():
    """
    Continuously segment, crop ROI and compare embeddings.
    """
    global cap
    frame_idx = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        frame_idx += 1
        # Salva frame corrente per debug
        if frame_idx % 10 == 1:  # salva solo ogni 10 frame per non saturare disco
            cv2.imwrite(f"debug_live_frame_{frame_idx}.jpg", frame)
            log(
                f"[DEBUG] Frame {frame_idx} salvato come debug_live_frame_{frame_idx}.jpg"
            )
        # 1) Run segmentation to get masks/boxes
        results = seg_model(frame, stream=False)
        # Assume results[0].masks.xy or boxes exist
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [[x1,y1,x2,y2],...]
        confs = results[0].boxes.conf.cpu().numpy()
        # Draw on frame
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Salva crop per debug
            crop_path = f"debug_crop_{frame_idx}_{i}.jpg"
            cv2.imwrite(crop_path, crop)
            log(f"[DEBUG] Crop ROI salvato: {crop_path}")
            emb_crop = get_embedding(crop)
            # Salva embedding crop per debug
            np.save(f"debug_crop_embedding_{frame_idx}_{i}.npy", emb_crop)
            log(
                f"[DEBUG] Embedding crop salvato: debug_crop_embedding_{frame_idx}_{i}.npy"
            )
            # Compute distances to prototypes
            d_front = np.linalg.norm(emb_crop - prototypes["front"])
            d_back = np.linalg.norm(emb_crop - prototypes["back"])
            side = "FRONTE" if d_front < d_back else "RETRO"
            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{side}: f{d_front:.2f} b{d_back:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        # Show frame
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    log("[DEBUG] Live detection terminata.")


# Thread control
stop_event = threading.Event()
model_thread = None


def start_live_detection():
    if prototypes["front"] is None:
        log("[ERROR] Prototipi non calcolati.")
        return
    stop_event.clear()
    global model_thread
    model_thread = threading.Thread(target=live_detection_loop, daemon=True)
    model_thread.start()
    log("[INFO] Live detection avviata. Premi 'q' sulla finestra per uscire.")


# ---------------------------------------
# 7) GUI Buttons
# ---------------------------------------
init_camera()
seg_model = YOLO("models/segmentation_best.pt")  # già addestrato su 100 buste

btn_front = ttk.Button(
    root, text="Scatta Fronte", command=lambda: capture_side("front")
)
btn_front.pack()
btn_back = ttk.Button(
    root, text="Scatta Retro", state=tk.DISABLED, command=lambda: capture_side("back")
)
btn_back.pack()
btn_live = ttk.Button(
    root, text="Avvia Live Detection", state=tk.DISABLED, command=start_live_detection
)
btn_live.pack()

root.mainloop()
