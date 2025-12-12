import requests

url = "http://127.0.0.1:5000/classify_plant"
payload = {
    "image_path": "E:\\6thSem\\GLYTCH'25\\GoogleMapDB\\GoogleVR4\\test\\Tree.jpg"  # Update this path to your local image file
}

response = requests.post(url, json=payload)
print("Plant Classification API response:")
print(response.json())