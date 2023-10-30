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
try:
    response = requests.post(url, json=patient)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'Dead':
            print('The patient is likely not going to survive.')
        else:
            print('The patient will survive.')
    else:
        print(f'Request failed with status code {response.status_code}')
except requests.exceptions.RequestException as e:
    print(f'Request error: {e}')