import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import threading
import cv2
import time
import os


def get_model_list():
    models_dir = "models"
    files = []
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pt"):
                files.append(f)
    return files


stop_event = threading.Event()
model_thread = None


def run_model(model_name, conf, fps, show, save):
    print(
        f"[DEBUG] Avvio modello {model_name} con conf={conf}, fps={fps}, show={show}, save={save}"
    )
    model = YOLO(f"models/{model_name}")
    results = model(
        source=0,
        show=show,
        conf=conf,
        save=save,
        stream=True,  # Per gestire FPS manualmente
    )
    delay = 1.0 / fps if fps > 0 else 0
    for r in results:
        if stop_event.is_set():
            print("[DEBUG] Stop richiesto, uscita dal loop di rilevamento.")
            break
        print(f"[DEBUG] Frame processato, conf={conf}")
        if delay > 0:
            time.sleep(delay)
        if not show:
            break  # Se non mostro, esco dopo il primo frame
    print("[DEBUG] Thread di rilevamento terminato.")


def start_model():
    global model_thread
    stop_event.clear()
    model_name = model_var.get()
    conf = float(conf_var.get())
    fps = int(fps_var.get())
    show = show_var.get()
    save = save_var.get()
    model_thread = threading.Thread(
        target=run_model, args=(model_name, conf, fps, show, save), daemon=True
    )
    model_thread.start()


def stop_model():
    stop_event.set()
    print("[DEBUG] Stop richiesto dall'utente.")
    # Attendi che il thread termini (opzionale)
    if model_thread and model_thread.is_alive():
        model_thread.join(timeout=2)
        print("[DEBUG] Thread fermato.")


def reload_all():
    stop_model()
    # Ricarica lista modelli
    new_list = get_model_list()
    model_combo["values"] = new_list
    if new_list:
        model_var.set(new_list[0])
    print("[DEBUG] Modelli ricaricati.")
    # Qui si potrebbe anche riaprire la telecamera se necessario


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

ttk.Label(mainframe, text="FPS:").grid(row=2, column=0, sticky=tk.W)
fps_var = tk.StringVar(value="10")
fps_entry = ttk.Entry(mainframe, textvariable=fps_var, width=5)
fps_entry.grid(row=2, column=1, sticky=tk.W)

show_var = tk.BooleanVar(value=True)
ttk.Checkbutton(mainframe, text="Show Window", variable=show_var).grid(
    row=3, column=0, sticky=tk.W
)

save_var = tk.BooleanVar(value=True)
ttk.Checkbutton(mainframe, text="Save Results", variable=save_var).grid(
    row=4, column=0, sticky=tk.W
)

ttk.Button(mainframe, text="Start Model", command=start_model).grid(
    row=5, column=0, pady=10, sticky=tk.W
)
ttk.Button(mainframe, text="Stop", command=stop_model).grid(
    row=5, column=1, pady=10, sticky=tk.E
)
ttk.Button(mainframe, text="Reload", command=reload_all).grid(
    row=5, column=2, pady=10, sticky=tk.E
)

# Debug area
debug_text = tk.Text(mainframe, height=8, width=40)
debug_text.grid(row=6, column=0, columnspan=2, pady=5)


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
