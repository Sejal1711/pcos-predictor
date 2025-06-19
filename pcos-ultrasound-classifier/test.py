import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
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
        self.root.title("PCOS Ultrasound Classifier")
        self.root.geometry("500x600")
        self.root.configure(bg="#f7f7f7")

        self.image_path = None
        self.image_label = Label(self.root, bg="#f7f7f7")
        self.image_label.pack(pady=10)

        self.select_button = Button(self.root, text="Select Ultrasound Image", command=self.select_image,
                                    bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.select_button.pack(pady=10)

        self.predict_button = Button(self.root, text="Predict", command=self.predict,
                                     bg="#2196F3", fg="white", font=("Arial", 12, "bold"), padx=20, pady=5)
        self.predict_button.pack(pady=10)

        self.result_label = Label(self.root, text="", font=("Arial", 14), bg="#f7f7f7")
        self.result_label.pack(pady=20)

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
            self.result_label.config(text="")  # Clear result on new image

    def predict(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            img_array = preprocess_image(self.image_path)
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

            if confidence < 0.5:
                result = "PCOS Detected"
                conf_display = 1.0 - confidence
            else:
                result = "Normal"
                conf_display = confidence

            self.result_label.config(
                text=f"Prediction: {result}\nConfidence: {conf_display:.4f}",
                fg="green" if result == "Normal" else "red"
            )
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

# === RUN APPLICATION ===
if __name__ == "__main__":
    root = tk.Tk()
    app = PCOSApp(root)
    root.mainloop()
