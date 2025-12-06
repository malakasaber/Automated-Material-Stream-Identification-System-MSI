import os
import cv2
import numpy as np

# ----------------------------
# (2) FUNCTIONS
# ----------------------------
def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img, filename))
    return images

def save_image(image, path):
    # Convert normalized image back to uint8 before saving
    img = ((image * np.array([0.229,0.224,0.225])) + np.array([0.485,0.456,0.406]))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))