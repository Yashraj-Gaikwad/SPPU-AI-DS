'''
Data Visualization using matplotlib

Problem Statement: Analyzing Air Quality Index (AQI) Trends in a City

Dataset: "City_Air_Quality.csv"

Description: The dataset contains information about air quality measurements in a specific
city over a period of time. It includes attributes such as date, time, pollutant levels (e.g., PM2.5,
PM10, CO), and the Air Quality Index (AQI) values. The goal is to use the matplotlib library
to create visualizations that effectively represent the AQI trends and patterns for different
pollutants in the city.

Tasks to Perform:
1. Import the "City_Air_Quality.csv" dataset.

2. Explore the dataset to understand its structure and content.

3. Identify the relevant variables for visualizing AQI trends, such as date, pollutant levels,
and AQI values.

4. Create line plots or time series plots to visualize the overall AQI trend over time.

5. Plot individual pollutant levels (e.g., PM2.5, PM10, CO) on separate line plots to
visualize their trends over time.

6. Use bar plots or stacked bar plots to compare the AQI values across different dates or
time periods.

7. Create box plots or violin plots to analyze the distribution of AQI values for different
pollutant categories.

8. Use scatter plots or bubble charts to explore the relationship between AQI values and
pollutant levels.

9. Customize the visualizations by adding labels, titles, legends, and appropriate color
schemes.

Reduced dataset

'''
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("air_quality_index.csv")

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Check for any missing values
print(data.isnull().sum())

# Step 4: Plotting the overall AQI distribution (no date involved)
plt.figure(figsize=(12, 6))
plt.hist(data['AQI'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of AQI Values')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Step 5: Plotting PM2.5 levels without date
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['PM2.5'], color='orange', label='PM2.5')
plt.title('PM2.5 Levels (Index Based)')
plt.xlabel('Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid()
plt.show()

# Step 6: Box plot for AQI distribution
plt.figure(figsize=(12, 6))
plt.boxplot(data['AQI'])
plt.title('Box Plot of AQI Values')
plt.ylabel('AQI')
plt.grid()
plt.show()

# Step 8: Scatter plot between PM2.5 and AQI without date
plt.figure(figsize=(12, 6))
plt.scatter(data['PM2.5'], data['AQI'], alpha=0.5)
plt.title('Scatter Plot of PM2.5 vs AQI')
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('AQI')
plt.grid()
plt.show()

# Example of customizing a plot further
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['AQI'], color='purple', linewidth=2)
ax.set_title('Customized Overall AQI Trend (Index Based)', fontsize=16)
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('AQI', fontsize=14)
ax.legend(['AQI Trend'])
ax.grid(True)
plt.tight_layout()
plt.show()


