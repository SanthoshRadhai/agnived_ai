import requests

url = "http://127.0.0.1:5000/panos"
payload = {
    "lat": 24.492786100000018,
    "lon": 77.34341670000003,
    "count": 3,
    "area_of_interest": 100,
    "min_distance": 20
}
response = requests.post(url, json=payload)
print("Panoramic API response:")
print(response.json())