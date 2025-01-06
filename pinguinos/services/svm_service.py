import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("pinguinos/models/svm_model.pkl")

@app.route('/predict/svm', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == "__main__":
    app.run(port=5000)
