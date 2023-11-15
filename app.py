import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pickle

app = Flask(__name__)

with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))



@app.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    # Ensure front-end input is the same as 'X_test' format and dimensionality from model.py
    int_features = [int(x) for x in request.form.values()]

    #Pre-processing and PCA process for input values - aligns it to X_test principal component


    final_features = [np.array(int_features)]
    prediction = knn_model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Credit Default Fraud Risk $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = knn_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

# Static Data Visualization & Overview - General Info. about Credit Card Fraud (Rosemarie & Josh O.)
# Input feature - Submit Excel/CSV file (Josh B. & Josh O.)
    # -inputs into our model and analyzes risks of each transaction.
    # -outputs excel file to local env.
# Financial Overview Webpage (optional)
    # -visualization, stats of spending habits

if __name__ == "__main__":
    app.run(debug=True)