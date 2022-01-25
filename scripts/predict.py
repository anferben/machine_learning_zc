import pickle
from flask import Flask
from flask import request
from flask import jsonify


output_file = f'model_C=1.0.bin'

with open(output_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('churn')


def churn_desicion (customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    return y_pred, churn


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    y_pred, churn = churn_desicion(customer=customer)
    
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)