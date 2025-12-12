import requests

url = "http://127.0.0.1:5000/detect_objects"
payload = {
    "image_path": r"E:\6thSem\GLYTCH'25\PipeLine3.0\GoogleVR4\test\Tree.jpg",
    "labels": ["tree", "bushes"]
}
response = requests.post(url, json=payload)
print("Object Detection API response:")
print(response.json())