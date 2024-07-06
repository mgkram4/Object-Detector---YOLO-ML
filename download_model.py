import requests
import os


def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


# URL for YOLOv7 weights
url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
filename = "yolov7.pt"

if not os.path.exists(filename):
    print(f"Downloading {filename}...")
    download_file(url, filename)
    print("Download completed.")
else:
    print(f"{filename} already exists.")

print(f"File size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")
