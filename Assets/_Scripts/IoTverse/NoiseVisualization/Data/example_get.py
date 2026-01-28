import requests
import urllib3
from pprint import pprint
import json

# Suppress SSL warnings when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_JEDI_URL = "https://djx.entlab.hr/m2m/trusted/data"

# First, get all available resources (no res parameter)
params = {
    "usr": "FER_Departments",
    "latestNCount":50,
    "res": "dipProj25_noise_detector1"  # Example resource to filter on
}
0
headers = {
    "Authorization": "PREAUTHENTICATED",
    "X-Requester-Id": "digiphy1",
    "X-Requester-Type": "domainApplication",
    "Accept": "application/vnd.ericsson.simple.output+json;version=1.0"
}

print("=== Fetching all available resources ===\n")
response = requests.get(
    DATA_JEDI_URL,
    params=params,
    headers=headers,
    verify=False
)

print(f"Status: {response.status_code}\n")
if response.status_code == 200:
    try:
        data = response.json()
        pprint(data)
        # Extract and display structure
        # if isinstance(data, list) and len(data) > 0:
        #     device = data[0]
        #     print("=== Available nodes/resources ===")
        #     for key in device.keys():
        #         if key != "deviceId":
        #             print(f"\nGateway/Node: {key}")
        #             if isinstance(device[key], dict):
        #                 for sensor_key in device[key].keys():
        #                     print(f"  └─ Sensor: {sensor_key}")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(response.text)
else:
    print(f"Error: {response.text}")

