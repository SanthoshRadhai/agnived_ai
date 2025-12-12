# import requests

# url = "http://127.0.0.1:5000/run_landcover_vegetation_and_panos"
# payload = {
#     # Landcover/vegetation AOI (will be forced > 3km by server)
#     "lon": 24.492786,
#     "lat": 77.343416,
#     "buffer_km": 3.1,
#     "date_start": "2024-10-01",
#     "date_end": "2024-11-15",
#     "scale": 10,
#     "cloud_cover_max": 20,
#     # Panos AOI (will be forced < 150m by server)
#     "panos_lat": 24.492786100000018,
#     "panos_lon": 77.34341670000003,
#     "panos_count": 3,
#     "panos_area_of_interest": 100,
#     "panos_min_distance": 20,
#     "panos_labels": ["tree", "bushes"]
# }
# response = requests.post(url, json=payload)
# print("Combined Landcover, Vegetation, and Panos API response:")
# try:
#     print(response.json())
# except Exception as e:
#     print("Error parsing response:", e)
#     print(response.text)


import requests

url = "http://127.0.0.1:5000/run_landcover_vegetation_and_panos"
payload = {
    # Landcover/vegetation AOI (will be forced > 3km by server)
    "lon": 77.303778100000018,
    "lat": 28.56027870000003,
    "buffer_km": 3.1,
    "date_start": "2024-10-01",
    "date_end": "2024-11-15",
    "scale": 10,
    "cloud_cover_max": 20,
    # Panos AOI (will be forced < 150m by server)
    "panos_lat": 24.492786100000018,
    "panos_lon": 77.34341670000003,
    "panos_count": 3,
    "panos_area_of_interest": 100,
    "panos_min_distance": 20,
    "panos_labels": ["tree", "bushes"]
}
response = requests.post(url, json=payload)
print("Combined Landcover, Vegetation, and Panos API response:")
try:
    print(response.json())
except Exception as e:
    print("Error parsing response:", e)
    print(response.text)
