import requests

url = "http://127.0.0.1:5000/detect_objects"
payload = {
    "image_path": "E:/6thSem/GLYTCH'25/GoogleMapDB/GoogleVR4/fl.jpg",
    "labels": ["tree", "bushes"]
}
response = requests.post(url, json=payload)
print("Object Detection API response:")
print(response.json())