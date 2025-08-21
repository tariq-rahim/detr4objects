import os
import torch
from torchvision.ops import nms
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === Configuration ===
MODEL_PATH = 'custom-model'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = r'dataset_path\test'  # Replace with your test images directory
OUTPUT_DIR = os.path.join(IMAGE_DIR, "output")
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load COCO-style label map from TEST_DATASET ===
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}

# === Load model and processor ===
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# === Get all image files in the directory ===
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(image_extensions)]

if not image_files:
    print(f"No images found in {IMAGE_DIR}")
else:
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"\nProcessing: {img_name}")

        # === Load and preprocess image ===
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        # === Inference ===
        with torch.no_grad():
            outputs = model(**inputs)

        # === Post-processing ===
        results = processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]], threshold=SCORE_THRESHOLD)
        result = results[0]

        if len(result["scores"]) == 0:
            print("No objects detected.")
            continue

        # === Extract predictions ===
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        # === Apply NMS ===
        keep_indices = nms(boxes, scores, NMS_THRESHOLD)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

        # === Print detections ===
        num_labels = len(id2label)
        colors = plt.colormaps['tab20'].resampled(num_labels)

        for score, label, box in zip(scores, labels, boxes):
            label_name = id2label.get(label.item(), f"Label_{label.item()}")
            print(f"Label: {label_name}, Score: {score.item():.3f}, Box: {box.tolist()}")

       # === Visualization (Corrected) ===
        # Get image dimensions (in pixels)
        img_width, img_height = image.size
        dpi = 100  # Choose a DPI value
        figsize = (img_width / dpi, img_height / dpi)

        # Create figure matching the image size
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
        ax.imshow(image)
        ax.set_axis_off()  # Remove axes completely

        # Draw bounding boxes and labels
        for score, label, box in zip(scores, labels, boxes):
            xmin = box[0].cpu().item()
            ymin = box[1].cpu().item()
            xmax = box[2].cpu().item()
            ymax = box[3].cpu().item()

            width, height = xmax - xmin, ymax - ymin
            label_name = id2label.get(label.item(), f"Label_{label.item()}")
            color = colors(label.item())

            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                    edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            ax.text(xmin, ymin - 10, f'{label_name}: {score:.2f}',
                    fontsize=12, color='white',
                    bbox=dict(facecolor=color, alpha=0.7, pad=2))

        # === Save output image (no whitespace) ===
        output_img_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_pred.jpg")
        fig.savefig(output_img_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved to: {output_img_path}")
