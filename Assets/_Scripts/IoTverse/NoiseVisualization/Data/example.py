from flask import Flask, request, jsonify
import requests, datetime

app = Flask(__name__)

DATA_JEDI_URL = "https://djx.entlab.hr/m2m/trusted/data"

@app.route("/sensors/temperature", methods=["POST"])
def receive_temperature():
    data = request.get_json()
    print(f"Received: {data}")

    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "resourceSpec": "temperature"
        },
        "contentNodes": [
            {
                "value": data["temperature"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    headers = {
        "Authorization": "PREAUTHENTICATED",
        "X-Requester-Id": "digiphy1",
        "X-Requester-Type": "domainApplication",
        "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers)
    print("Data Jedi response:", r.status_code, r.text)
    return jsonify({"status": "ok", "platform_code": r.status_code})


@app.route("/sensors/noisedetector", methods=["POST"])
def receive_noise():
    data = request.get_json()
    print(f"Received noise: {data}")

    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "resource": "dipProj25_noise_detector"
        },
        "contentNodes": [
            {
                "value": data["noise"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    headers = {
        "Authorization": "PREAUTHENTICATED",
        "X-Requester-Id": "digiphy1",
        "X-Requester-Type": "domainApplication",
        "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)