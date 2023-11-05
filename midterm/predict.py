import pickle
from flask import Flask, jsonify, request

with open('model.bin', 'rb') as f:
   model, dv = pickle.load(f)


app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.get_json()

        required_fields = ['age', 't_stage', 'n_stage', '6th_stage', 'differentiate', 'grade',
       'tumor_size', 'estrogen_status', 'progesterone_status',
       'regional_node_examined', 'regional_node_positive', 'survival_months',
       'size_classification', 'lymph_node_positivity_%']

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
