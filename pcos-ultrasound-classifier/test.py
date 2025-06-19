import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox, Frame, Text, Scrollbar
import os

# === MODEL SETUP ===
MODEL_PATH = "pcos_ultrasound_model.h5"
IMG_HEIGHT, IMG_WIDTH = 224, 224

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# === IMAGE PREPROCESSING ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === GUI APPLICATION ===
class PCOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCOS Predictor")
        self.root.geometry("720x880")
        self.root.configure(bg="#f0f8ff")

        # === Introduction Text ===
        intro = (
            "🔍 Welcome to the PCOS Predictor!\n\n"
            "This AI-powered app analyzes your ultrasound image to detect signs of "
            "Polycystic Ovary Syndrome (PCOS). It gives not just predictions but also helpful "
            "tips if PCOS is detected. Upload your image and let the AI do its magic!"
        )
        Label(root, text=intro, wraplength=680, bg="#f0f8ff",
              font=("Arial", 11), justify="left").pack(padx=20, pady=(10, 5))

        # === Condensed PCOS Info ===
        info = (
            "💡 PCOS is a hormonal condition that affects women. Symptoms may include irregular periods, "
            "acne, weight gain, or cysts on the ovaries. It can impact fertility but is manageable "
            "through diet, exercise, stress control, and medical care."
        )
        Label(root, text=info, wraplength=680, bg="#e0f0ff", justify="left",
              font=("Arial", 10), relief="solid", padx=10, pady=10).pack(padx=20, pady=10)

        # === Image Preview ===
        self.image_path = None
        self.image_label = Label(self.root, bg="#f0f8ff")
        self.image_label.pack(pady=10)

        # === Buttons ===
        btn_frame = Frame(root, bg="#f0f8ff")
        btn_frame.pack(pady=5)
        Button(btn_frame, text="📂 Select Image", command=self.select_image,
               bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), padx=10).pack(side="left", padx=10)
        Button(btn_frame, text="🔍 Predict", command=self.predict,
               bg="#2196F3", fg="white", font=("Arial", 11, "bold"), padx=20).pack(side="left", padx=10)

        # === Prediction Output with Scrollbar ===
        result_frame = Frame(self.root)
        result_frame.pack(padx=20, pady=15, fill="both", expand=True)

        self.result_box = Text(result_frame, height=12, wrap="word", font=("Arial", 11),
                               bg="#ffffff", relief="solid", padx=10, pady=5)
        self.result_box.configure(state="disabled")
        self.result_box.pack(side="left", fill="both", expand=True)

        scrollbar = Scrollbar(result_frame, command=self.result_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_box.config(yscrollcommand=scrollbar.set)

    def select_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        path = filedialog.askopenfilename(title="Select an Ultrasound Image", filetypes=filetypes)
        if path:
            self.image_path = path
            img = Image.open(path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            self.result_box.configure(state="normal")
            self.result_box.delete("1.0", tk.END)
            self.result_box.configure(state="disabled")

    def predict(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            img_array = preprocess_image(self.image_path)
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

            if confidence < 0.5:
                result = "⚠️ PCOS Detected"
                conf_display = 1.0 - confidence
                motivation = (
                    "\n\nTips:\n✔ Eat a balanced diet\n✔ Exercise daily\n✔ Reduce stress\n"
                    "✔ Visit your doctor\n\n💪 You can manage PCOS. You're not alone!"
                )
            else:
                result = "✅ Normal"
                conf_display = confidence
                motivation = "\n\n🎉 Your scan looks normal! Stay healthy and active. 💚"

            output = f"{result}\nConfidence: {conf_display:.4f}{motivation}"
            self.result_box.configure(state="normal")
            self.result_box.delete("1.0", tk.END)
            self.result_box.insert("1.0", output)
            self.result_box.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

# === RUN ===
if __name__ == "__main__":
    root = tk.Tk()
    app = PCOSApp(root)
    root.mainloop()
