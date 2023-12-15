#import necessary libraries
import os
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import pickle
import traceback
import sqlite3
from flask import g
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests import Session
from sklearn.preprocessing import OneHotEncoder, LabelEncoder




# specify HTML static folder and the template folder
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['DEBUG'] = True

#sign session cookies to protect the session data from being manipulated by users.
# app.secret_key = '22803002003' 
app.secret_key = os.urandom(24)

# Load the pre-trained model
with open('final_refined_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False)


# Connect to SQLite database
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('flood_prediction_database.db', detect_types=sqlite3.PARSE_DECLTYPES)
        
        # Create a table to store predictions
        with app.app_context():
            cursor = db.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city TEXT,
                    prediction_result INTEGER,
                    prediction_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create a table to store preprocessed data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preprocessed_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city TEXT,
                    timezone INTEGER,
                    lat REAL,
                    lon REAL,
                    temp REAL,
                    visibility INTEGER,
                    dew_point REAL,
                    feels_like REAL,
                    temp_min REAL,
                    temp_max REAL,
                    pressure INTEGER,
                    humidity INTEGER,
                    wind_speed REAL,
                    wind_gust REAL,
                    rain REAL,
                    clouds_all REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            db.commit()
        
    return db

# Close the database connection when the app is closed
@app.teardown_appcontext
def close_connection(exception=None):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


open_weather_API_key = os.getenv('OPEN_WEATHER_API_KEY')

# Dictionary to store city-coordinate mappings and url to their weather forcast
city_coordinates = {
    'Freetown': {'lat': 8.484, 'lon': -13.2299, 'url':'https://openweathermap.org/city/2409306'},
    'Bo': {'lat': 7.9647, 'lon': -11.7383, 'url':'https://openweathermap.org/city/2410048'},
    'Kenema': {'lat': 7.8767, 'lon': -11.1875, 'url':'https://openweathermap.org/city/2407790'}, 
    'Koidu': {'lat': 8.4167, 'lon': -10.8333, 'url':'https://openweathermap.org/city/2407660'}, 
    'Makeni': {'lat': 8.8833, 'lon': -12.05, 'url':'https://openweathermap.org/city/2406407'}, 
    'Magburaka': {'lat': 8.7167, 'lon': -11.95, 'url':'https://openweathermap.org/city/2406576'}, 
    'Port Loko': {'lat': 8.7667, 'lon': -12.7833, 'url':'https://openweathermap.org/city/2404433'}, 
    'Bonthe': {'lat': 7.5264, 'lon': -12.505}, 'url':'https://openweathermap.org/city/2409914', 
    'Moyamba': {'lat': 8.1667, 'lon': -12.4333, 'url':'https://openweathermap.org/city/2405013'}, 
    'Moyamba Junction': {'lat': 8.3262, 'lon': -12.2153, 'url':'https://openweathermap.org/city/2402715'}, 
    'Kabala': {'lat': 9.5833, 'lon': -11.55, 'url':'https://openweathermap.org/city/2408329'}, 
    'Falaba': {'lat': 9.8555, 'lon': -11.3211, 'url':'https://openweathermap.org/city/2410093'}, 
    'Mile 91': {'lat': 8.4659, 'lon': -12.2116, 'url':'https://openweathermap.org/city/2402715'}, 
    'Lunsar': {'lat': 8.6833, 'lon': -12.5333, 'url':'https://openweathermap.org/city/2406916'}, 
    'Kambia': {'lat': 9.1167, 'lon': -12.9167, 'url':'https://openweathermap.org/city/2408088'}, 
    'Taiama': {'lat': 8.1999, 'lon': -12.0614, 'url':'https://openweathermap.org/city/2407089'}, 
    'Njala': {'lat': 8.1136, 'lon': -12.0741, 'url':'https://openweathermap.org/city/2407089'}, 
    'Mokonde': {'lat': 7.7627, 'lon': -12.1681, 'url':'https://openweathermap.org/city/2403896'}, 
    'Pujehun': {'lat': 7.3581, 'lon': -11.7208, 'url':'https://openweathermap.org/city/2571039'}, 
    'Kailahun': {'lat': 8.2833, 'lon': -10.5667, 'url':'https://openweathermap.org/city/2408250'}, 
    'Lungi': {'lat': 9.2212, 'lon': -12.6831, 'url':'https://openweathermap.org/city/2407262'}, 
    'Mattru': {'lat': 7.6244, 'lon': -11.8332, 'url':'https://openweathermap.org/city/2409215'}, 
    'Masiaka': {'lat': 8.5807, 'lon': -12.1418, 'url':'https://openweathermap.org/city/2402715'}, 
    'Bumpe': {'lat': 7.8919, 'lon': -11.9025, 'url':'https://openweathermap.org/city/2409823'}, 
    'Waterloo': {'lat': 8.3383, 'lon': -13.0719, 'url':'https://openweathermap.org/city/2403094'}, 
    'Daru': {'lat': 7.9919, 'lon': -10.8406, 'url':'https://openweathermap.org/city/2409663'}, 
}

#render app to html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the selected city from the form
        city = request.form.get('city')
        if not city or not city.strip():
            return jsonify(error='Invalid city name.')
        
        # Get coordinates for the selected city
        coordinates = city_coordinates.get(city)

        # Make API call to get weather data
        if coordinates:
            lat, lon, city_url = coordinates['lat'], coordinates['lon'], coordinates['url']
            api_url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={open_weather_API_key}'
            
            session = Session()
            retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            response = session.get(api_url)
            
            
            if response.status_code == 200:
                # The API returns JSON data
                data = response.json()

                # Preprocess the data
                input_features = preprocess_data(data)
                
                print("Result of preprocess_data:")
                print(input_features)
            
                # Make predictions using the model
                prediction = model.predict(input_features)

                # Convert the prediction to 0 or 1 in order to avoid BLOB data in database
                binary_prediction = int(prediction[0])

                # Determine the message based on the prediction
                if binary_prediction == 1:
                    message = f"There is a likelihood that it will flood in {city} today! Check the weather <a href='{city_url}' target='_blank'>here</a>."
                else:
                    message = f"There is no likelihood of flooding in {city} at the today! Check the weather <a href='{city_url}' target='_blank'>here</a>."
                    
                # Store the prediction result in the database
                store_prediction_in_database(city, binary_prediction, message)

                # Store the preprocessed data in the database
                store_preprocessed_data_in_database(city, input_features)

                # Return the JSON response
                return jsonify(result=message)  
            else:
                return jsonify(error=f"Failed to retrieve data from API. Status Code: {response.status_code}") 

    return jsonify(error='City coordinates not found.')


def store_prediction_in_database(city, prediction, message):
    # Store the prediction result in the database
    cursor = get_db().cursor()
    cursor.execute('''
        INSERT INTO predictions (city, prediction_result, prediction_message)
        VALUES (?, ?, ?)
    ''', (city, prediction, message))
    get_db().commit()

def store_preprocessed_data_in_database(city, input_features):
    # Store the preprocessed data in the database
    cursor = get_db().cursor()
    cursor.execute('''
        INSERT INTO preprocessed_data (city, timezone, lat, lon, temp, visibility, dew_point, feels_like, temp_min, temp_max, pressure, humidity, wind_speed, wind_gust, rain, clouds_all)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        city,
        input_features['timezone'].values[0].astype(float),
        input_features['lat'].values[0],
        input_features['lon'].values[0],
        input_features['temp'].values[0],
        input_features['visibility'].values[0].astype(float),
        input_features['dew_point'].values[0],
        input_features['feels_like'].values[0],
        input_features['temp_min'].values[0].astype(float),
        input_features['temp_max'].values[0].astype(float),
        input_features['pressure'].values[0].astype(float),
        input_features['humidity'].values[0].astype(float),
        input_features['wind_speed'].values[0],
        input_features['wind_gust'].values[0],
        input_features['rain'].values[0].astype(float),
        input_features['clouds_all'].values[0].astype(float),
    ))
    get_db().commit()

# preprocess data
def preprocess_data(data):
    current_data = data.get('current', {})
    daily_data = data.get('daily', [{}])[0] 
    timezone = data.get('timezone')
    latitude = data.get('lat')
    longitude = data.get('lon')
    temperature = daily_data.get('temp', {}).get('day', 0)
    visibility = current_data.get('visibility')
    dew_point = daily_data.get('dew_point', {})
    feels_like = daily_data.get('feels_like', {}).get('day', 0)
    temp_min = daily_data.get('temp', {}).get('min', 0)
    temp_max = daily_data.get('temp', {}).get('max', 0)
    pressure = daily_data.get('pressure', {})
    humidity = daily_data.get('humidity', {})
    wind_speed = daily_data.get('wind_speed', {})
    wind_gust = daily_data.get('wind_gust', {})
    rain = daily_data.get('rain', {}).get('rain', 0)  # Access the 'rain' attribute
    clouds_all = daily_data.get('clouds', {})
    
     # Map 'timezone' to numerical label
    timezone_mapping = {
        'Africa/Freetown': 0,
        # Add other timezone mappings as needed
    }
    
    timezone_label = timezone_mapping.get(timezone, -1)  # Use -1 if the timezone is not found in the mapping
    
     # Handle missing or unexpected timezone values
    if timezone_label == -1:
        print(f"Unexpected timezone value: {timezone}")
        # Replace timezone_label with 0
        timezone_label = 0
        
    # Convert 'timezone' to integer
    timezone_label = int(timezone_label)

    # Create a DataFrame with the extracted features
    input_features = pd.DataFrame({
        'timezone': [timezone_label],
        'lat': [latitude],
        'lon': [longitude],
        'temp': [temperature],
        'visibility': [visibility],
        'dew_point': [dew_point],
        'feels_like': [feels_like],
        'temp_min': [temp_min],
        'temp_max': [temp_max],
        'pressure': [pressure],
        'humidity': [humidity],
        'wind_speed': [wind_speed],
        'wind_gust': [wind_gust],
        'rain': [rain],
        'clouds_all': [clouds_all]
    })

    # Handle missing values
    input_features['wind_gust'].fillna(0, inplace=True)
   # input_features['rain'].fillna(0, inplace=True)
    input_features["visibility"].fillna(input_features["visibility"].median(), inplace=True)
    input_features.fillna(0, inplace=True)
    
    # Handle missing values using SimpleImputer
    # imputer = SimpleImputer(strategy='mean')  
    # input_features = pd.DataFrame(imputer.fit_transform(input_features), columns=input_features.columns)

    return input_features

@app.errorhandler(Exception)
def handle_error(e):
    traceback.print_exc()  
    return jsonify(error=str(e)), 500
 
 
#  @app.after_request
#  def add_security_headers(response):
#      response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
#      return response
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
