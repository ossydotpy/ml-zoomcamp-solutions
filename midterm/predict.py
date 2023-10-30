import pickle
import xgboost as xgb
from flask import Flask, jsonify, request

with open('model.bin', 'rb') as f:
   model, dv = pickle.load(f)


app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.get_json()

        required_fields = [
            "6th_stage", "a_stage", "age_group", "differentiate", "estrogen_status",
            "grade", "lymph_node_positivity_%", "marital_status", "n_stage",
            "progesterone_status", "race", "regional_node_examined", "regional_node_positive",
            "size_classification", "survival_months", "t_stage"
        ]

        missing_fields = [field for field in required_fields if field not in features]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        if not isinstance(features["lymph_node_positivity_%"], (int, float)) or features["lymph_node_positivity_%"] < 0:
            return jsonify({"error": "Invalid 'lymph_node_positivity_%' value"}), 400


        X = dv.transform([features])
        y_pred = model.predict(X)

        prediction = 'Dead' if y_pred[0] == 1 else 'Survive'

        response = {
            "status": prediction
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=4041)
