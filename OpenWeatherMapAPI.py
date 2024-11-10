'''
Interacting with Web APIs

Problem Statement: Analyzing Weather Data from OpenWeatherMap API

Dataset: Weather data retrieved from OpenWeatherMap API

Description: The goal is to interact with the OpenWeatherMap API to retrieve weather data
for a specific location and perform data modeling and visualization to analyze weather
patterns over time.

Tasks to Perform:
1. Register and obtain API key from OpenWeatherMap.

2. Interact with the OpenWeatherMap API using the API key to retrieve weather data for
a specific location.

3. Extract relevant weather attributes such as temperature, humidity, wind speed, and
precipitation from the API response.

4. Clean and preprocess the retrieved data, handling missing values or inconsistent
formats.

5. Perform data modeling to analyze weather patterns, such as calculating average
temperature, maximum/minimum values, or trends over time.

6. Visualize the weather data using appropriate plots, such as line charts, bar plots, or
scatter plots, to represent temperature changes, precipitation levels, or wind speed
variations.

7. Apply data aggregation techniques to summarize weather statistics by specific time
periods (e.g., daily, monthly, seasonal).

8. Incorporate geographical information, if available, to create maps or geospatial
visualizations representing weather patterns across different locations.

9. Explore and visualize relationships between weather attributes, such as temperature
and humidity, using correlation plots or heatmaps.


viva questions
1) 
'''

# for making API calls
import requests
# for data handling
import pandas as pd
# for data visualization
import matplotlib.pyplot as plt
# for data visualization
import seaborn as sns

# Step 1: Set your API key and base URL
API_KEY = '21205b3ed0b78a7545e5d8b585bb87fd'
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Step 2: Function to get weather data for a specific location
def get_weather_data(city):
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'  # Use 'imperial' for Fahrenheit
    }
    # Error Handling
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Example city to retrieve weather data for
cities = ['Pune', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai']
weather_data = get_weather_data(cities)

# Step 3: Extract relevant weather attributes
# Checks HTTP Status code 200 means valid response 
# Collect weather data for multiple cities
weather_data_list = []
for city in cities:
    weather_data = get_weather_data(city)
    if weather_data and weather_data['cod'] == 200:
        main_data = weather_data['main']
        wind_data = weather_data['wind']
        
        extracted_data = {
            'City': city,
            'Temperature': main_data['temp'],
            'Humidity': main_data['humidity'],
            'Wind Speed': wind_data['speed'],
            'Pressure': main_data['pressure'],
            'Weather Description': weather_data['weather'][0]['description']
        }
        weather_data_list.append(extracted_data)

# Convert to DataFrame for easier manipulation
weather_df = pd.DataFrame(weather_data_list)

# Print the collected data
print(weather_df)

# Step 4: Clean and preprocess the data (if needed)
# Handle missing values or format inconsistencies if necessary.

# Step 5: Perform data modeling (e.g., calculating averages)
average_Temp = weather_df['Temperature'].mean()
average_Humidity = weather_df['Humidity'].mean()
average_Wind = weather_df['Wind Speed'].mean()
average_Pressure = weather_df['Pressure'].mean()

# Print the extracted data
print(f"Average Temperature in {cities}: {average_Temp:.2f} °C")
print(f"Average Humidity in {cities}: {average_Humidity:} %")
print(f"Average Wind Speed in {cities}: {average_Wind:.2f} m/s")
print(f"Average Pressure in {cities}: {average_Pressure:} hPa")


# Step 6: Visualize the Weather Data

# Bar plot for current temperature
plt.figure(figsize=(10, 5))
sns.barplot(x='City', y='Temperature', data=weather_df)
plt.title('Current Temperature')
plt.ylabel('Temperature (°C)')
# Loop through the df for temp
for index, value in enumerate(weather_df['Temperature']):
    # show the text 0.5 pixels on top of bar chart in center
    plt.text(index, value + 0.5, f'{value:.1f}', ha='center')
plt.show()

# Scatter plot for Wind Speed vs Temperature
plt.figure(figsize=(10, 5))
sns.scatterplot(data=weather_df, x='Temperature', y='Wind Speed', hue='City', s=100)
plt.title('Wind Speed vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Wind Speed (m/s)')
plt.show()

# Step 7: Data Aggregation (Example)
# For demonstration purposes, let's create a mock dataset for daily averages.
mock_daily_data = {
    'Date': pd.date_range(start='2023-01-01', periods=7),
    'Temperature': [10, 12, 15, 14, 13, 11, 16],
    'Humidity': [70, 65, 80, 75, 73, 68, 60],
    'Wind Speed': [5.0, 4.5, 3.0, 2.5, 4.0, 3.5, 6.0]
}
daily_weather_df = pd.DataFrame(mock_daily_data)

# Aggregating daily average temperature
daily_avg_temp = daily_weather_df.groupby('Date').mean().reset_index()

# Line plot for Daily Average Temperature
plt.figure(figsize=(10, 5))
sns.lineplot(data=daily_avg_temp, x='Date', y='Temperature', marker='o')
plt.title('Daily Average Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Average Temperature (°C)')
plt.xticks(rotation=0)
plt.show()

# Step 8: Incorporate Geographical Information (Optional)
# For geographical visualization you can use Folium or Geopandas.
# Here’s a simple example of how to create a map using Folium:
import folium

map_center = [20.5937, 78.9629] # Center of India
m = folium.Map(location=map_center, zoom_start=5)

for _, row in weather_df.iterrows():
    folium.Marker(
        location=[20.5937 + _ * 0.1, 78.9629 + _ * 0.1], # Dummy coordinates for illustration
        popup=f"{row['City']}: {row['Temperature']} °C",
    ).add_to(m)

m.save("weather_map.html")
print("Geographical map saved as weather_map.html")

# Step 9: Explore Relationships Between Weather Attributes
# Select only numeric columns for correlation
numeric_weather_df = weather_df.select_dtypes(include=['float64', 'int64'])

# Check if there are any numeric columns
if not numeric_weather_df.empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_weather_df.corr(), annot=True)
    plt.title('Correlation between Weather Attributes')
    plt.show()
else:
    print("No numeric data available for correlation.")
