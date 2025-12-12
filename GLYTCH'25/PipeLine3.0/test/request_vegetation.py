import requests

url = "http://127.0.0.1:5000/run_vegetation"
payload = {
    "lon": 77.303778,
    "lat": 28.560278,
    "buffer_km": 3.0,
    # Optionally specify mask_path if needed
    "mask_path": r"E:\6thSem\GLYTCH'25\PipeLine3.0\LandcoverResults\vegetation_mask.tif"
}
response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response:", response.json())
