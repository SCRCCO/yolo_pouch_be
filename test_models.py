import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
import threading
import os
from PIL import Image, ImageTk


stop_event = threading.Event()
model_thread = None
selected_image_path = None
result_image_path = None
selected_img_tk = None
result_img_tk = None


def get_model_list():
    models_dir = "models"
    files = []
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pt"):
                files.append(f)
    return files


def update_image_preview(img_path, label_widget, maxsize=(320, 240)):
    global selected_img_tk, result_img_tk
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path)
        img.thumbnail(maxsize)
        tk_img = ImageTk.PhotoImage(img)
        label_widget.config(image=tk_img)
        # Salva riferimento per evitare garbage collection
        if label_widget == image_preview_label:
            selected_img_tk = tk_img
        else:
            result_img_tk = tk_img
    else:
        label_widget.config(image="")


def run_model_on_image(model_name, conf, save, image_path):
    print(
        f"[DEBUG] Avvio inferenza su immagine {image_path} con modello {model_name}, conf={conf}, save={save}"
    )
    model = YOLO(f"models/{model_name}")
    results = model(
        source=image_path,
        conf=conf,
        save=save,
        show=False,
    )
    global result_image_path
    result_image_path = None

    # Cerca la directory di salvataggio dal log ultralytics
    save_dir = None
    if hasattr(results, "save_dir") and results.save_dir:
        save_dir = results.save_dir
    else:
        # fallback: cerca l'ultima cartella runs/detect/predict*
        runs_dir = os.path.join("runs", "detect")
        if os.path.isdir(runs_dir):
            predict_dirs = [
                os.path.join(runs_dir, d)
                for d in os.listdir(runs_dir)
                if d.startswith("predict")
            ]
            if predict_dirs:
                save_dir = max(predict_dirs, key=os.path.getmtime)

    if save and save_dir and os.path.isdir(save_dir):
        # Cerca il file risultato più recente
        result_files = [
            os.path.join(save_dir, fname)
            for fname in os.listdir(save_dir)
            if fname.lower().endswith((".jpg", ".png"))
        ]
        if result_files:
            result_image_path = max(result_files, key=os.path.getmtime)

    print(f"[DEBUG] Inferenza completata. Immagine risultato: {result_image_path}")
    root.after(0, lambda: update_image_preview(result_image_path, result_preview_label))


def show_result_image():
    # Non più necessario, preview aggiornata direttamente nell'interfaccia
    pass


def select_image():
    global selected_image_path
    filetypes = [("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    path = filedialog.askopenfilename(title="Seleziona immagine", filetypes=filetypes)
    if path:
        selected_image_path = path
        img_name = os.path.basename(path)
        image_label.config(text=f"Immagine selezionata: {img_name}")
        print(f"[DEBUG] Immagine selezionata: {path}")
        update_image_preview(selected_image_path, image_preview_label)


def start_inference():
    if not selected_image_path:
        messagebox.showwarning("Attenzione", "Seleziona prima un'immagine.")
        return
    model_name = model_var.get()
    conf = float(conf_var.get())
    save = save_var.get()
    threading.Thread(
        target=run_model_on_image,
        args=(model_name, conf, save, selected_image_path),
        daemon=True,
    ).start()


def reload_all():
    # Ricarica lista modelli
    new_list = get_model_list()
    model_combo["values"] = new_list
    if new_list:
        model_var.set(new_list[0])
    print("[DEBUG] Modelli ricaricati.")


root = tk.Tk()
root.title("YOLO Segmentation GUI")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Modello selezionabile
ttk.Label(mainframe, text="Modello:").grid(row=0, column=0, sticky=tk.W)
model_var = tk.StringVar()
model_list = get_model_list()
model_combo = ttk.Combobox(
    mainframe, textvariable=model_var, values=model_list, state="readonly", width=28
)
if model_list:
    model_var.set(model_list[0])
model_combo.grid(row=0, column=1, sticky=tk.W)

ttk.Label(mainframe, text="Confidence threshold:").grid(row=1, column=0, sticky=tk.W)
conf_var = tk.StringVar(value="0.4")
conf_entry = ttk.Entry(mainframe, textvariable=conf_var, width=5)
conf_entry.grid(row=1, column=1, sticky=tk.W)

save_var = tk.BooleanVar(value=True)
ttk.Checkbutton(mainframe, text="Save Results", variable=save_var).grid(
    row=2, column=0, sticky=tk.W
)

ttk.Button(mainframe, text="Seleziona Immagine", command=select_image).grid(
    row=3, column=0, pady=10, sticky=tk.W
)
image_label = ttk.Label(mainframe, text="Nessuna immagine selezionata")
image_label.grid(row=3, column=1, sticky=tk.W)

# Anteprima immagine caricata
image_preview_label = tk.Label(mainframe)
image_preview_label.grid(row=6, column=0, pady=5, padx=5)
ttk.Label(mainframe, text="Anteprima immagine caricata").grid(
    row=5, column=0, sticky=tk.W
)

# Anteprima risultato inferenza
result_preview_label = tk.Label(mainframe)
result_preview_label.grid(row=6, column=1, pady=5, padx=5)
ttk.Label(mainframe, text="Risultato inferenza").grid(row=5, column=1, sticky=tk.W)

ttk.Button(mainframe, text="Inferisci", command=start_inference).grid(
    row=4, column=0, pady=10, sticky=tk.W
)
ttk.Button(mainframe, text="Reload Modelli", command=reload_all).grid(
    row=4, column=1, pady=10, sticky=tk.E
)

# Debug area
debug_text = tk.Text(mainframe, height=8, width=40)
debug_text.grid(row=7, column=0, columnspan=2, pady=5)


def log_debug(msg):
    debug_text.insert(tk.END, msg + "\n")
    debug_text.see(tk.END)


# Redirect print to debug area
import sys


class StdoutRedirector:
    def write(self, s):
        log_debug(s.strip())

    def flush(self):
        pass


sys.stdout = StdoutRedirector()

root.mainloop()
