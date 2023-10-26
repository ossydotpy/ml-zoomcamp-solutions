import requests

url = 'http://0.0.0.0:4041/predict'

patient =  {
    "6th_stage": "iia",
    "a_stage": "regional",
    "age_group": "senior",
    "differentiate": "well_differentiated",
    "estrogen_status": "positive",
    "grade": "1",
    "lymph_node_positivity_%": 50.0,
    "marital_status": "married",
    "n_stage": "n1",
    "progesterone_status": "negative",
    "race": "white",
    "regional_node_examined": 2,
    "regional_node_positive": 1,
    "size_classification": "Small",
    "survival_months": 102,
    "t_stage": "t1"
  }
request = requests.post(url,json=patient)

if request.status_code == 200:
    response = request.json()
    if response['status'] == 'Dead':
        print('patient is likely not going to survive.')
    else:
        print('Patient will survive.')