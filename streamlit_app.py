import tkinter as tk
from tkinter import filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import time
from detector import LicensePlateDetector

class PlateDashboard:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Dashboard Reconnaissance de plaques")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e2f")

        self.detector = LicensePlateDetector(model_path)
        self.camera_running = False
        self.cap = None
        self.last_frame = None

        # ----- Widgets -----
        # Cadre des boutons
        btn_frame = tk.Frame(root, bg="#2d2d44")
        btn_frame.pack(pady=10)

        self.btn_image = tk.Button(btn_frame, text="📁 Charger une image", command=self.load_image,
                                   font=("Arial", 12), bg="#4c9aff", fg="white", padx=15, pady=5)
        self.btn_image.grid(row=0, column=0, padx=10)

        self.btn_camera = tk.Button(btn_frame, text="🎥 Démarrer caméra", command=self.start_camera,
                                    font=("Arial", 12), bg="#4c9aff", fg="white", padx=15, pady=5)
        self.btn_camera.grid(row=0, column=1, padx=10)

        self.btn_stop = tk.Button(btn_frame, text="⏹️ Arrêter caméra", command=self.stop_camera,
                                  font=("Arial", 12), bg="#ff6b6b", fg="white", padx=15, pady=5, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=2, padx=10)

        # Zone d'affichage vidéo / image
        self.video_label = tk.Label(root, bg="#0a0a14", relief="sunken")
        self.video_label.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

        # Zone de texte pour les résultats
        self.log_area = scrolledtext.ScrolledText(root, height=8, font=("Consolas", 10),
                                                   bg="#2d2d44", fg="#00ffcc", insertbackground="white")
        self.log_area.pack(pady=10, padx=10, fill=tk.X)
        self.log("Dashboard prêt. Choisissez une image ou démarrez la caméra.")

    def log(self, message):
        """Ajoute un message dans la zone de texte."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_area.see(tk.END)

    def show_image(self, cv_img):
        """Convertit une image OpenCV (BGR) en PhotoImage et l'affiche."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Redimension pour l'affichage (max 800x600)
        max_w, max_h = 800, 600
        scale = min(max_w/w, max_h/h, 1.0)
        new_w, new_h = int(w*scale), int(h*scale)
        rgb = cv2.resize(rgb, (new_w, new_h))
        img_pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

    def load_image(self):
        """Ouvre une boîte de dialogue pour choisir une image."""
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not filepath:
            return
        self.log(f"Chargement de l'image : {filepath}")
        img = cv2.imread(filepath)
        if img is None:
            self.log("Erreur : impossible de lire l'image.")
            return
        annotated_img, plates = self.detector.detect_and_recognize(img)
        self.show_image(annotated_img)
        if plates:
            self.log(f"🔹 Plaques trouvées : {', '.join(plates)}")
        else:
            self.log("❌ Aucune plaque détectée sur cette image.")

    def start_camera(self):
        """Lance le flux webcam dans un thread séparé."""
        if self.camera_running:
            self.log("Caméra déjà active.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("Impossible d'ouvrir la webcam.")
            return
        self.camera_running = True
        self.btn_camera.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.log("Caméra démarrée. Détection en temps réel...")
        self.update_camera()

    def update_camera(self):
        """Lecture et traitement d'une frame (appelée récursivement)."""
        if not self.camera_running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.log("Perte du flux caméra.")
            self.stop_camera()
            return
        annotated_frame, plates = self.detector.detect_and_recognize(frame)
        self.show_image(annotated_frame)
        if plates:
            self.log(f"📷 {', '.join(plates)}")
        # Rappel après 30 ms (~30 FPS)
        self.root.after(30, self.update_camera)

    def stop_camera(self):
        """Arrête la caméra."""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.log("Caméra arrêtée.")
        # Effacer l'affichage
        self.video_label.config(image='')
        self.video_label.image = None

    def on_close(self):
        """Fermeture propre."""
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # METTEZ ICI LE CHEMIN VERS VOTRE MODÈLE YOLO (Plaques)
    MODEL_PATH = "models/votre_modele.pt"   # <--- À modifier
    app = PlateDashboard(root, MODEL_PATH)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()