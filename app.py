# Dependencies
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pickle
import logging

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

    # Create a dictionary to store values for each target column
    # Create a DataFrame
    flat_data = [item[0] for item in input_features]

    # Create a DataFrame
    X_val = pd.DataFrame([flat_data], columns=target_columns)
    print('X_val')
    print(X_val)
    # Drop the 'category' column
    #Validation dataset
    dummy_categories = pd.get_dummies(data=X_val['category'])

    X_newval = X_val.drop(['category'], axis=1)

    for category in dummy_categories:
        X_newval[category] = dummy_categories[category]


    print("X_newval")
    print(X_newval)

    #%%
    # Dimensionality Reduction

    #Validation Dataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    

    X_normalized = X_newval.values
    X_normalized = StandardScaler().fit_transform(X_normalized)
    new_column = np.zeros(X_normalized.shape[0])
    X_normalized = np.column_stack((X_normalized, new_column))
    print("X_normalized")
    print(X_normalized)

    
    # pca = PCA(n_components=6)
    # principal_components = pca.fit_transform(X_normalized)
    # int_features_new = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2',
    #                                                                     'principal component 3', 'principal component 4',
    #                                                                   'principal component 5', 'principal component 6'])
    int_features_new = pd.DataFrame(data = X_normalized, columns = ['principal component 1', 'principal component 2',
                                                                        'principal component 3', 'principal component 4',
                                                                      'principal component 5', 'principal component 6'])
    # Example: Return the processed features
    return int_features_new


# Flask App Pathways
@app.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    print("Received POST request")
    
    data = request.form.to_dict(flat=False)

    # Assuming your input features are passed as JSON in the request
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

# Static Data Visualization & Overview - General Info. about Credit Card Fraud (Rosemarie & Josh O.)
# Input feature - Submit Excel/CSV file (Josh B. & Josh O.)
    # -inputs into our model and analyzes risks of each transaction.
    # -outputs excel file to local env.
# Financial Overview Webpage (optional)
    # -visualization, stats of spending habits

if __name__ == "__main__":
    app.run(debug=True)