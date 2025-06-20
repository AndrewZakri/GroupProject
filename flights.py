import pandas as pd
import plotly.express as px
import folium
import numpy as np
import math
import calendar
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
import streamlit as st

#Title of the app
st.title("Airline A Dashboard")
st.write("Analysis of Airline A data to determine potential upgrade of existing fleet")

#Data reading

df = pd.read_csv("Airports_P 1.csv")
dt = pd.read_csv("Airports_T 1.csv")
dd = pd.read_csv("Airports_D.csv")

monthly_passengers = dt.groupby('Fly_date')['Passengers'].sum().reset_index()
monthly_passengers['Fly_date'] = pd.to_datetime(monthly_passengers['Fly_date'])
monthly_passengers = monthly_passengers.sort_values(by='Fly_date')
monthly_passengers['Rolling_Avg'] = monthly_passengers['Passengers'].rolling(window=3).mean()

#Create a line chart
fig1 = px.line(
    monthly_passengers,
    x='Fly_date',
    y=['Passengers', 'Rolling_Avg'],
    title='Monthly Passengers with Rolling Average (3 months)',
    color_discrete_map={
        'Passengers': 'blue',
        'Rolling_Avg': 'green'
    }
)
fig1.show(0)

monthly_flights = dt.groupby('Fly_date')['Flights'].sum().reset_index()
monthly_flights['Fly_date'] = pd.to_datetime(monthly_flights['Fly_date'])
monthly_flights = monthly_flights.sort_values(by='Fly_date')
monthly_flights['Rolling_Avg'] = monthly_flights['Flights'].rolling(window=3).mean()

#Create a second line chart
fig2 = px.line(
    monthly_flights,
    x='Fly_date',
    y=['Flights', 'Rolling_Avg'],
    title='Monthly Flights with Rolling Average (3 months)',
    labels={'Fly_date': 'Year', 'Flights': 'Number of Flights'},
    color_discrete_map={
        'Flights': 'blue',
        'Rolling_Avg': 'green'
    }
)
fig2.show(0)

# ----------------- Forecasting Passengers ------------------
# Preprocess the data for monthly passengers
monthly_passengers = dt.groupby('Fly_date')['Passengers'].sum().reset_index()
monthly_passengers['Fly_date'] = pd.to_datetime(monthly_passengers['Fly_date'])
monthly_passengers = monthly_passengers.sort_values(by='Fly_date')
monthly_passengers.set_index('Fly_date', inplace=True)

# Fit the Holt-Winters Exponential Smoothing Model for Passengers (multiplicative trend and seasonality)
model_passengers = ExponentialSmoothing(monthly_passengers['Passengers'],
                                         trend='mul',
                                         seasonal='mul',
                                         damped_trend=True,
                                         seasonal_periods=12,
                                         initialization_method='estimated')

fit_passengers = model_passengers.fit()
forecast_passengers = fit_passengers.forecast(24)

# Plot for Passengers Forecast
plt.figure(figsize=(10, 5))
plt.plot(monthly_passengers['Passengers'], label='Observed (Passengers)', color='blue')
plt.plot(forecast_passengers.index, forecast_passengers, label='Forecast (Passengers)', linestyle='--', color='green')
plt.legend(loc='lower right')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.title('Holt-Winters Forecast for Passengers')

# Display the plots in Streamlit
st.plotly_chart(fig1)

st.plotly_chart(fig2)
st.subheader("Airline A Passenger Forecast")
st.pyplot(plt)
