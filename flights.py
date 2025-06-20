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

#Forcasting the time series for number of flights
monthly_flights = dt.groupby('Fly_date')['Flights'].sum().reset_index()
monthly_flights['Fly_date'] = pd.to_datetime(monthly_flights['Fly_date'])
monthly_flights = monthly_flights.sort_values(by='Fly_date')
monthly_flights.set_index('Fly_date', inplace=True)

model = ExponentialSmoothing(monthly_flights['Flights'],
                              trend='mul',
                              seasonal='mul',
                              damped_trend=True,
                              seasonal_periods=12,
                              initialization_method='estimated')


fit = model.fit()
forecast = fit.forecast(24)

plt.figure(figsize=(10, 5))
plt.plot(monthly_flights['Flights'], label='Observed')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Number of Flights')
plt.title('Airline A Flights Forecast')
plt.show()

#Forcasting the time series for number of passengers
monthly_passengers= dt.groupby('Fly_date')['Passengers'].sum().reset_index()
monthly_passengers['Fly_date'] = pd.to_datetime(monthly_passengers['Fly_date'])
monthly_passengers = monthly_passengers.sort_values(by='Fly_date')
monthly_passengers.set_index('Fly_date', inplace=True)

model = ExponentialSmoothing(monthly_passengers['Passengers'],
                              trend='mul',
                              seasonal='mul',
                              damped_trend=True,
                              seasonal_periods=12,
                              initialization_method='estimated')


fit = model.fit()
forecast = fit.forecast(24)

plt2.figure(figsize=(10, 5))
plt2.plot(monthly_passengers['Passengers'], label='Observed')
plt2.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt2.legend(loc='lower right')
plt2.xlabel('Year')
plt2.ylabel('Number of Passengers')
plt2.title('Holt-Winters Forecast')
plt2.show()

# Display the plots in Streamlit
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.pyplot(plt)
st.pyplot(plt2)
