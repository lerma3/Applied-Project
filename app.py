# Static Data Visualization & Overview - General Info. about Credit Card Fraud (Rosemarie & Josh O.)
# Input feature - Submit Excel/CSV file (Josh B. & Josh O.)
    # -inputs into our model and analyzes risks of each transaction.
    # -outputs excel file to local env.
# Financial Overview Webpage (optional)
    # -visualization, stats of spending habits

#%%
# Dependencies - General
import pandas as pd
import numpy as np
from flask import Flask, request, flash, jsonify, render_template, url_for, redirect
import pickle
import logging

# Dependency - Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Dependency - Excel File upload
import os
from werkzeug.utils import secure_filename

# Flask App - Initialization
app = Flask(__name__)
app.logger.setLevel(logging.INFO)  # Set the log level as needed

    
# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to validate file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls', 'csv'}

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


#____________________________________________________________---

# Flask App Pathways
@app.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')

# Single Prediction Route
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

# Excel Upload - Mass Predict Feature
@app.route('/mass_predict',methods=['GET', 'POST'])
def mass_predict():
    if request.method == 'POST':
        print("Received Excel POST request")

        if 'file' not in request.files:
            return redirect(url_for('mass_predict'))

        file = request.files['file']

        # Check if a file was selected
        if file.filename == '':
            return redirect(url_for('mass_predict'))

        # Check if the file has an allowed extension
        if not allowed_file(file.filename):
            return redirect(url_for('mass_predict'))
        
        try:
            # Save the file
            file.save(file.filename)

            # Read the Excel or CSV file into a Pandas DataFrame
            input_data = pd.read_excel(file)  # Update this based on your actual data format

            # Process the input_data 
            print(input_data)

            # Return the processed data or redirect to another page
            return render_template('mass_upload.html', data=input_data.to_html())
        except Exception as e:
            print(f'Error processing file: {str(e)}', 'error')
        # finally:
        #     # Clean up: remove the uploaded file
        #     os.remove(file.filename)

    # Render the initial form
    return render_template('mass_upload.html')
    
#Data Visualization Page
@app.route('/visual',methods=['GET', 'POST'])
def visual():
    # Sample data for the chart
    labels = ['January', 'February', 'March', 'April', 'May']
    data = [10, 20, 15, 25, 30]

    # Pass data to the template
    return render_template('visual.html', labels=labels, data=data)




if __name__ == "__main__":
    app.run(debug=True)
# %%
