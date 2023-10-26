import pickle
import xgboost as xgb
from flask import Flask, request

with open('dv.bin', 'rb') as f:
   dv = pickle.load(f)

model = xgb.XGBClassifier()
model.load_model('model.json')

app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.get_json()

    X = dv.transform([features])
    y_pred = model.predict(X)

    prediction = 'Dead' if y_pred[0] == 1 else 'Survive'

    response = {
        "status": str(prediction)
    }

    return response


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=4041)
