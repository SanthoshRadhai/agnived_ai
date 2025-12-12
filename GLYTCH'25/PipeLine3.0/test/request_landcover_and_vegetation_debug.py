import requests

url = "http://127.0.0.1:5000/run_landcover_and_vegetation"
payload = {
    "lon": 77.303778,
    "lat": 28.560278,
    "buffer_km": 3.0,
    "date_start": "2024-10-01",
    "date_end": "2024-11-15",
    "scale": 10,
    "cloud_cover_max": 20
}
response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Raw Response Text:")
print(response.text)
try:
    print("Parsed JSON Response:")
    print(response.json())
except Exception as e:
    print("Error parsing JSON:", e)
