from flask import Flask, request, jsonify
import requests, datetime

app = Flask(__name__)

DATA_JEDI_URL = "https://djx.entlab.hr/m2m/trusted/data"
headers = {
        "Authorization": "PREAUTHENTICATED",
        "X-Requester-Id": "digiphy1",
        "X-Requester-Type": "domainApplication",
        "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
    }

@app.route("/sensors/temperature/1", methods=["POST"])
def receive_temperature1():
    data = request.get_json()
    print(f"Received: {data}")


    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "res": "dipProj25_temperature1"
        },
        "contentNodes": [
            {
                "value": data["temperature"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/temperature/2", methods=["POST"])
def receive_temperature2():
    data = request.get_json()
    print(f"Received: {data}")


    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "res": "dipProj25_temperature2"
        },
        "contentNodes": [
            {
                "value": data["temperature"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/temperature/3", methods=["POST"])
def receive_temperature3():
    data = request.get_json()
    print(f"Received: {data}")


    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "res": "dipProj25_temperature3"
        },
        "contentNodes": [
            {
                "value": data["temperature"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/temperature/4", methods=["POST"])
def receive_temperature4():
    data = request.get_json()
    print(f"Received: {data}")


    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "res": "dipProj25_temperature4"
        },
        "contentNodes": [
            {
                "value": data["temperature"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/noisedetector/1", methods=["POST"])
def receive_noise1():
    data = request.get_json()
    print(f"Received noise: {data}")

    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "resource": "dipProj25_noise_detector1"
        },
        "contentNodes": [
            {
                "value": data["noise"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/noisedetector/2", methods=["POST"])
def receive_noise2():
    data = request.get_json()
    print(f"Received noise: {data}")

    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "resource": "dipProj25_noise_detector2"
        },
        "contentNodes": [
            {
                "value": data["noise"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/noisedetector/3", methods=["POST"])
def receive_noise3():
    data = request.get_json()
    print(f"Received noise: {data}")

    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "resource": "dipProj25_noise_detector3"
        },
        "contentNodes": [
            {
                "value": data["noise"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})

@app.route("/sensors/noisedetector/4", methods=["POST"])
def receive_noise4():
    data = request.get_json()
    print(f"Received noise: {data}")

    payload = {
        "source": {
            "operator": "Group104",
            "domainApplication": "Group104_domain",
            "user": "FER_Departments",
            "resource": "dipProj25_noise_detector4"
        },
        "contentNodes": [
            {
                "value": data["noise"],
                "time": datetime.datetime.now(datetime.UTC).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)