from PIL import Image
import os

def validate_images(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify this is a readable image
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image found and removed: {file_path}")
                os.remove(file_path)

# Change this path to your test folder path
validate_images(r"C:\Users\NEHA\Downloads\pcos\data\test")
