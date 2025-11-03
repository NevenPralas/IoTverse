# IoTverse

## Deep Learning

### How to setup

Create a new python virtual environment: `python -m venv venv`  
Make sure to start the environment (unix): `source venv/bin/activate`  
Make sure to start the environment (Windows): `\venv\Scripts\Activate`  
Install necessary dependencies: `pip install -r requirements.txt`  

### How to use

Initialize the database by running: `python database.py`  
Add dummy data by running: `python populate_db.py`  
View the time-temperature graph by running: `streamlit run dashboard.py`  
Predict next temperature value by running: `python model.py`  

