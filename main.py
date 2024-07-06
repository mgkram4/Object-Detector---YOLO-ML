import torch
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def plot_one_box(box, img, color=(255, 0, 0), label=None, line_thickness=None):
    """Draws one bounding box on the image."""
    x1, y1, x2, y2 = [int(x) for x in box]
    thickness = line_thickness or int(round(0.002 * max(img.shape[0:2])))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_size = 0.5
        cv2.putText(
            img,
            label,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            color,
            thickness,
        )


def make_writable(img):
    """Ensure image is writable."""
    if not img.flags.writeable:
        return np.copy(img)
    return img


def display(results, show=True, save_path=None):
    """Display and optionally save the results."""
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in results.names]
    for img, pred in zip(results.imgs, results.pred):
        img = make_writable(img)
        img = np.ascontiguousarray(img)
        for *box, conf, cls in reversed(pred):
            label = f"{results.names[int(cls)]} {conf:.2f}"
            plot_one_box(box, img, label=label, color=colors[int(cls) % len(colors)])

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Annotated image saved to {save_path}")

        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.show()


def main():
    # Load the model
    print("Loading YOLO v7 model...")
    model = torch.hub.load("WongKinYiu/yolov7", "custom", "yolov7.pt")
    print("Model loaded successfully.")

    # Sample Image URL
    BASE_URL = "https://github.com/ultralytics/yolov5/raw/master/data/images/"
    image_url = BASE_URL + "zidane.jpg"

    # Load the image from URL
    print(f"Downloading image from {image_url}")
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_np = np.array(image)
    print("Image downloaded and converted to numpy array.")

    # Inference
    print("Running inference...")
    results = model(image_np)

    # Display and save results
    display(results, save_path="annotated_image.jpg")

    # Print the results
    print("Detection Results:")
    print(results.xyxy[0])

    # Print all class names
    print("\nClass Names:")
    print(model.names)

    print("Processing complete.")


if __name__ == "__main__":
    main()
