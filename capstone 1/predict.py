import pickle
import xgboost as xgb

from flask import Flask, jsonify, request


VECTORIZER_FILE = 'models/vectorizer-0.74.bin'
MODEL_FILE = 'models/model-0.74.json'


with open(VECTORIZER_FILE, 'rb') as f:
    dv = pickle.load(f)


model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)


app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        records = request.get_json()

        if not isinstance(records, list):
            return jsonify({"error": "Input must be a list of records"}), 400

        predictions = []

        for record in records:
            required_fields = ['sex', 'age', 'height', 'weight', 'waistline', 'sight_left',
               'sight_right', 'hear_left', 'hear_right', 'sbp', 'dbp', 'blds',
               'tot_chole', 'hdl_chole', 'ldl_chole', 'triglyceride', 'hemoglobin',
               'urine_protein', 'serum_creatinine', 'sgot_ast', 'sgot_alt',
               'gamma_gtp', 'smk_stat_type_cd']

            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            X = dv.transform([record])
            y_pred = model.predict(X)
            prediction = 'Drinker' if y_pred[0] == 1 else 'Smoker'
            predictions.append({"status": prediction})

        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=4041)
