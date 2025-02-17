from flask import Flask, render_template, request,jsonify
import requests
import numpy
import joblib
from io import BytesIO
import sklearn

app = Flask(__name__)
  # Ensure model is available
model_path='model.pkl'
model=joblib.load(model_path)
# Load the model
# iris_model = joblib.load(MODEL_PATH)



@app.route('/predict', methods=['GET', 'POST'])
def iris():
    if request.method == 'POST':
            sepal_length = float(request.json['sepal_length'])
            sepal_width = float(request.json['sepal_width'])
            petal_length = float(request.json['petal_length'])
            petal_width = float(request.json['petal_width'])
            ans = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

            ans_name = ["Setosa", "Versicolor", "Virginica"][ans[0]]
            return jsonify({"answer":ans_name})


if __name__=="__main__":
     app.run(debug=True)
    
