# Landing page - General Info. about Credit Card Fraud
    # -about section: pull the purpose from project proposal
    # -contact info.
# Move single query to a different html template.
    # -repurpose index.html.
# Excel Upload Webpage - Submit Excel/CSV file (Josh B. & Josh O.)
    # -embed template for user to download locally and use to upload (add some instruction notes on webpage.)
    # -inputs into our model and analyzes risks of each transaction.
    # -outputs excel file to local env.
# Visualization Webpage
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
import zipcodes
import requests

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
    int_features_new = pca.fit_transform(X_normalized)
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

@app.route('/single_query', methods=('GET', 'POST'))
def single_query():
    return render_template('single_query.html')

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
            # print(input_data)

            state_abbreviations = {
                'Alabama': 'AL',
                'Montana': 'MT',
                'Alaska': 'AK',
                'Nebraska': 'NE',
                'Arizona': 'AZ',
                'Nevada': 'NV',
                'Arkansas': 'AR',
                'New Hampshire': 'NH',
                'California': 'CA',
                'New Jersey': 'NJ',
                'Colorado': 'CO',
                'New Mexico': 'NM',
                'Connecticut': 'CT',
                'New York': 'NY',
                'Delaware': 'DE',
                'District of Columbia': 'DC',
                'North Carolina': 'NC',
                'Florida': 'FL',
                'North Dakota': 'ND',
                'Georgia': 'GA',
                'Ohio': 'OH',
                'Hawaii': 'HI',
                'Oklahoma': 'OK',
                'Idaho': 'ID',
                'Oregon': 'OR',
                'Illinois': 'IL',
                'Pennsylvania': 'PA',
                'Indiana': 'IN',
                'Rhode Island': 'RI',
                'Iowa': 'IA',
                'South Carolina': 'SC',
                'Kansas': 'KS',
                'South Dakota': 'SD',
                'Kentucky': 'KY',
                'Tennessee': 'TN',
                'Louisiana': 'LA',
                'Texas': 'TX',
                'Maine': 'ME',
                'Utah': 'UT',
                'Maryland': 'MD',
                'Vermont': 'VT',
                'Massachusetts': 'MA',
                'Virginia': 'VA',
                'Michigan': 'MI',
                'Washington': 'WA',
                'Minnesota': 'MN',
                'West Virginia': 'WV',
                'Mississippi': 'MS',
                'Wisconsin': 'WI',
                'Missouri': 'MO',
                'Wyoming': 'WY',
            }

            def retrieve_population_data():
                api_key = 'ba2122268817b7d343412a8ac9317b5618bda67a'
                url = f"https://api.census.gov/data/2019/pep/population?get=NAME,POP&for=place:*&in=state:*&key={api_key}"
                response = requests.get(url)
                col_names = ['city/state', 'population', 'state_id', 'city_id']
                pop_df = pd.DataFrame(columns=col_names, data=response.json()[1:])
                #expanding city/state to two separate columns
                pop_df[['city', 'state']] = pop_df['city/state'].str.split(', ', 1, expand=True)
                pop_df.drop('city/state', axis=1, inplace=True)
                #making city lower case
                pop_df['city'] = pop_df['city'].str.lower()
                #mapping state abbreviations
                pop_df['state'] = pop_df['state'].map(lambda x: state_abbreviations.get(x, x))
                #removing 'town', 'city', 'village' from each city name
                pop_df['city'].replace('\scity|town|village', '', regex=True, inplace=True)
                return pop_df

            def preprocessing_data(df, populations):
                # converting timestamps to DateTime and to unix
                df['unix_time'] = pd.to_datetime(df['Transaction Date/Timestamp'])
                df['unix_time'] = pd.to_numeric(df['unix_time'])

                # get dummy category fields
                dummy_categories = pd.get_dummies(data=df['Transaction Category'])
                for category in dummy_categories:
                    df[category] = dummy_categories[category].astype(int)

                # mapping each zip code to city/state and pulling population
                # creating a place holder column
                df['city_pop'] = 0
                n = 0
                for zip in df['Transaction Zip Code']:
                    try:
                        state = zipcodes.matching(str(zip))[0]['state'].upper()
                        city = zipcodes.matching(str(zip))[0]['city'].lower()
                        population = pop_df[(pop_df['city'] == city) & (pop_df['state'] == state)]['population'].values
                        if population.size == 0:
                        # if population is not found, enter the mean population for the state
                            df['city_pop'][n] = int((pop_df[(pop_df['state'] == state)]['population'].astype(int).mean()))
                        else:
                            df['city_pop'][n] = str(population)[2:-2]
                    except:
                        df['city_pop'][n] = 0
                    n += 1
                
                df = df.drop(['Transaction Category', 'Transaction Zip Code', 'Transaction Date/Timestamp'], axis = 1)
                return df

            def pca(clean_df):
                df_normalized = clean_df.values
                df_normalized = StandardScaler().fit_transform(df_normalized)

                pca = PCA(n_components=4)
                principal_components = pca.fit_transform(df_normalized)
                principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2',
                                                                                'principal component 3', 'principal component 4'])
                return principal_df

            def predict(pca_df, model):
                predictions = model.predict(pca_df)
                predictions_df = pd.DataFrame(predictions)
                predictions_df.replace([0, 1], ['no', 'yes'], inplace=True)
                return predictions_df

            def user_output(predictions_df, user_input):
                predictions_df.replace([0, 1], ['no', 'yes'], inplace=True)
                user_output = user_input.copy()
                user_output['potential_fraud'] = predictions_df[0]
                return user_output

            populations = retrieve_population_data()
            print('input_data')
            print(input_data)
            clean_df = preprocessing_data(input_data, populations)
            print('clean_df')
            print(clean_df)
            pca_df = pca(clean_df)
            print('pca_df')
            print(pca_df)
            model = pickle.load(open('knn_model_2.pkl', 'rb'))
            predictions_df = predict(pca_df, model)
            print('predictions_df')
            print(predictions_df)
            user_output_file = user_output(predictions_df, input_data)
            user_output_file.to_excel('FraudPredictionResults.xlsx')


            # Return the processed data or redirect to another page
            return render_template('mass_upload.html', data=user_output_file.to_html())
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
