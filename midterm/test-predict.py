import requests

url = 'http://0.0.0.0:4041/predict'

patient =  {
    "age": 64,
    "t_stage": "t3",
    "n_stage": "n3",
    "6th_stage": "iiic",
    "differentiate": "poorly_differentiated",
    "grade": "3",
    "tumor_size": 55,
    "estrogen_status": "positive",
    "progesterone_status": "negative",
    "regional_node_examined": 12,
    "regional_node_positive": 11,
    "survival_months": 7,
    "size_classification": "Medium",
    "lymph_node_positivity_%": 91.6666666667
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