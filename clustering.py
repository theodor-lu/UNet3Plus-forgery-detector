import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import shutil

# --------------------------
# CONFIGURATION
# --------------------------
IMAGE_FOLDER = "data/non-rectangular/images"
OUTPUT_FOLDER = "data/clustered_forged_non-rect"
NUM_CLUSTERS = 4  # e.g., microscopy, CT, MRI, X-ray
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# PREPROCESSING
# --------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# --------------------------
# LOAD MODEL
# --------------------------
model = efficientnet_b0(pretrained=True)
model.classifier = torch.nn.Identity()  # remove classification head
model = model.to(DEVICE)
model.eval()

# --------------------------
# IMAGE CONVERSION HELPER
# --------------------------
def convert_to_rgb(img):
    """
    Convert image to RGB format.
    - Grayscale (L) -> RGB (duplicate channels)
    - RGBA -> RGB (drop alpha channel)
    - RGB -> RGB (no change)
    """
    if img.mode == 'RGB':
        return img
    elif img.mode == 'L':  # Grayscale
        return img.convert('RGB')
    elif img.mode == 'RGBA':  # RGBA
        # Create a white background and paste the image
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        return rgb_img
    else:
        # For other modes (P, CMYK, etc.), convert to RGB
        return img.convert('RGB')

# --------------------------
# LOAD IMAGES AND EXTRACT FEATURES
# --------------------------
file_names = []
features_list = []

# Supported image extensions
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

for fname in os.listdir(IMAGE_FOLDER):
    # Only process image files
    if not any(fname.lower().endswith(ext) for ext in image_extensions):
        continue
    
    path = os.path.join(IMAGE_FOLDER, fname)
    try:
        img = Image.open(path)
        img = convert_to_rgb(img)  # Handle grayscale and RGBA
    except Exception as e:
        print(f"Skipping {fname}: {e}")
        continue
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(x)  # shape: [1, 1280]
    features_list.append(features.cpu().numpy().flatten())
    file_names.append(fname)

features_matrix = np.stack(features_list)  # N x 1280

# --------------------------
# DIMENSIONALITY REDUCTION
# --------------------------
pca = PCA(n_components=50)
features_reduced = pca.fit_transform(features_matrix)

# --------------------------
# CLUSTERING
# --------------------------
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features_reduced)

# --------------------------
# CREATE OUTPUT FOLDERS
# --------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for i in range(NUM_CLUSTERS):
    os.makedirs(os.path.join(OUTPUT_FOLDER, f"cluster_{i}"), exist_ok=True)

# --------------------------
# COPY IMAGES TO CLUSTER FOLDERS
# --------------------------
for fname, label in zip(file_names, labels):
    src = os.path.join(IMAGE_FOLDER, fname)
    dst = os.path.join(OUTPUT_FOLDER, f"cluster_{label}", fname)
    shutil.copy2(src, dst)  # copy2 preserves metadata and timestamps

print(f"Clustering complete! Processed {len(file_names)} images into {NUM_CLUSTERS} clusters.")
print(f"Results saved to: {OUTPUT_FOLDER}")
print(f"Cluster distribution:")
for i in range(NUM_CLUSTERS):
    count = np.sum(labels == i)
    print(f"  Cluster {i}: {count} images")
