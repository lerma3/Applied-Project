import requests

url = 'http://localhost:5000/results'

# Create a dictionary with the user input
data = {
    'name': name,
    'address': address,
    'dob': dob,
    'gender': gender,
    'amt': amt,
    'zip': zip_code,
    'category': category
}

# Make a POST request
r = requests.post(url, json=data)