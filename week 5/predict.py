import pickle
from flask import Flask, request

with open('model2.bin', 'rb') as f:
    model = pickle.load(f)

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)


app = Flask('app')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    customer_vector = dv.transform([customer])
    prediction = model.predict_proba(customer_vector)[0,1]

    churn = prediction >= 0.5

    result = {
        'churn': bool(churn),
        'probability': float(prediction)
    }

    return result

if __name__== "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
