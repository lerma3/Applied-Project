# Static Data Visualization & Overview - General Info. about Credit Card Fraud (Rosemarie & Josh O.)
# Input feature - Submit Excel/CSV file (Josh B. & Josh O.)
    # -inputs into our model and analyzes risks of each transaction.
    # -outputs excel file to local env.
# Financial Overview Webpage (optional)
    # -visualization, stats of spending habits

#%%
# Dependencies
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pickle
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Flask App - Initialization
app = Flask(__name__)
app.logger.setLevel(logging.INFO)  # Set the log level as needed


# Load knn-model
with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))

# Pre-processing function
def preprocess_input(input_features):
    # Your preprocessing logic here
    # Pre-Processing
    # Data Modeling Techniques
    
    #restructuring data for model
    target_columns =['amt', 'city_pop', 'zip', 'category', 'unix_time']
    #target_columns = ['first', 'last', 'dob', 'gender', 'street', 'city', 'state', 'zip', 'job', 'amt']

    # Create flat data 
    flat_data = [item[0] for item in input_features]

    # Create a DataFrame
    X_val = pd.DataFrame([flat_data], columns=target_columns)
    print('X_val')
    print(X_val)

    # One-hot encoding
    dummy_categories = pd.get_dummies(data=X_val['category'])

    # Drop the original 'category' column and concatenate the one-hot encoded columns
    # X_newval = X_val.drop(['category'], axis=1)

    # for category in dummy_categories:
    #     X_newval[category] = dummy_categories[category]
    X_newval = pd.concat([X_val.drop(['category'], axis=1), dummy_categories], axis=1)
    
    X_newval = X_newval.apply(pd.to_numeric, errors='coerce')

    print("\nX_newval:")
    print(X_newval)
    
    # Dimensionality Reduction
    X_normalized = StandardScaler().fit_transform(X_newval)

    print("\nX_normalized:")
    print(X_normalized)

    n_samples, n_features = X_normalized.shape
    print("\nIncoming Features to PCA from X_normalized:")
    print(n_features)

    pca = PCA(n_components=1)
    int_features_new = pca.transform(X_normalized)
    print('\nPCA transformation:')
    print(int_features_new)
    return int_features_new


# Flask App Pathways
@app.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    print("Received POST request")
    
    data = request.form.to_dict(flat=False)

    # Assuming input features
    # 'amt', 'city_pop', 'zip', 'category', 'unix_time'
    input_features = [
        data['amt'],
        data['city_pop'],
        data['zip'],
        data['category'],
        data['unix_time'],
    ]

    print(f'Input Features: {input_features}')

    # print("Received POST request")
    # # Ensure front-end input is the same as 'X_test' format and dimensionality from model.py
    # int_features = [int(request.form['name']), 
    #                 int(request.form['address']), 
    #                 int(request.form['dob']), 
    #                 int(request.form['gender']),
    #                 float(request.form['amt']),
    #                 int(request.form['zip']),
    #                 request.form['category']]

    # print(f'Input Features: {int_features}')

    # Pre-processing and PCA process for input values - aligns it to X_test principal component
    final_features = preprocess_input(input_features)

    print(f'Final Features: {final_features}')

    # Prediction
    prediction = knn_model.predict([final_features][0])

    # Output the predicted result to the console
    print(f'Predicted Result: {prediction}')

    #output = round(prediction[0], 2)

    return jsonify({'prediction': prediction.tolist()})
    #return render_template('index.html', prediction_text='Credit Default Fraud Risk $ {}'.format(output))

# @app.route('/results',methods=['POST'])
# def results():
#     try:
#         data = request.get_json(force=True)
#         input_values = list(data.values())
#         # Preprocessing and PCA process for input values
#         final_features = preprocess_input(input_values)

#         prediction = knn_model.predict([final_features])
#         output = prediction[0]

#         app.logger.info(f'Prediction Result: {output}')

#         return jsonify({'prediction': output})
#     except Exception as e:
#         app.logger.error(f'Error: {e}')
#         # Log or handle the exception as needed
#         return jsonify({'error': str(e)})



if __name__ == "__main__":
    app.run(debug=True)