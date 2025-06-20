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

monthly_flights = dt.groupby('Fly_date')['Flights'].sum().reset_index()
monthly_flights['Fly_date'] = pd.to_datetime(monthly_flights['Fly_date'])
monthly_flights = monthly_flights.sort_values(by='Fly_date')
monthly_flights['Rolling_Avg'] = monthly_flights['Flights'].rolling(window=3).mean()

#Create a second line chart
fig2 = px.line(
    monthly_flights,
    x='Fly_date',
    y=['Flights', 'Rolling_Avg'],
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
plt.title('Holt-Winters Forecast')
plt.show()

# Display the plots in Streamlit
st.subheader("Airline A Monthly Flights with Rolling Average (3 months)")
st.plotly_chart(fig2)
st.subheader("Airline A Flights Forecast")
st.pyplot(plt)
plt.clf()

monthly_passengers = dt.groupby('Fly_date')['Passengers'].sum().reset_index()
monthly_passengers['Fly_date'] = pd.to_datetime(monthly_passengers['Fly_date'])
monthly_passengers = monthly_passengers.sort_values(by='Fly_date')
monthly_passengers['Rolling_Avg'] = monthly_passengers['Passengers'].rolling(window=3).mean()

#Create a line chart
fig1 = px.line(
    monthly_passengers,
    x='Fly_date',
    y=['Passengers', 'Rolling_Avg'],
    color_discrete_map={
        'Passengers': 'blue',
        'Rolling_Avg': 'green'
    }
)
fig1.show(0)

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
st.subheader("Airline A Monthly Passengers with Rolling Average (3 months)")
st.plotly_chart(fig1)
st.subheader("Airline A Passenger Forecast")
st.pyplot(plt)
plt.clf()

#Make a histogram of Distance from the data dratio, dont need if statement
plt.figure(figsize=(10, 6))
plt.hist(dd['Distance'], bins=50, edgecolor='black')
plt.title('Histogram of Distance (1990-2008)')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.xticks(ticks=range(0, int(dd['Distance'].max()) + 500, 500))
plt.grid(axis='y', alpha=0.75)
plt.show()

st.subheader("Airline A Flight Distance Frequency")
st.pyplot(plt)
plt.clf()

data = dt
data['Fly_date'] = pd.to_datetime(data['Fly_date'])
data['month'] = data['Fly_date'].dt.month.apply(lambda x: calendar.month_abbr[x])

Passengers = data.groupby('month')['Passengers'].sum().reset_index()
Seats = data.groupby('month')['Seats'].sum().reset_index()
Flights = data.groupby('month')['Flights'].sum().reset_index()

# prompt: plot the bargraph of Passengers, Seats, Flights by the months in the 3 figures of the same fig

# Order months correctly
month_order = [calendar.month_abbr[i] for i in range(1, 13)]
Passengers['month'] = pd.Categorical(Passengers['month'], categories=month_order, ordered=True)
Seats['month'] = pd.Categorical(Seats['month'], categories=month_order, ordered=True)
Flights['month'] = pd.Categorical(Flights['month'], categories=month_order, ordered=True)

Passengers = Passengers.sort_values('month')
Seats = Seats.sort_values('month')
Flights = Flights.sort_values('month')

fig, axes = plt.subplots(1, 3, figsize=(15, 10))

axes[0].bar(Passengers['month'], Passengers['Passengers'])
axes[0].set_title('Total Passengers by Months (1990-2008)')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total Passengers')

axes[1].bar(Seats['month'], Seats['Seats'], color='orange')
axes[1].set_title('Total Seats by Months (1990-2008)')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Total Seats')

axes[2].bar(Flights['month'], Flights['Flights'], color='green')
axes[2].set_title('Total Flights by Months (1990-2008)')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Total Flights')

plt.tight_layout()
plt.show()

st.subheader("Airline A Passengers, Seats & Flights by Calendar Month")
st.pyplot(fig)
plt.clf()

Utilization = Passengers['Passengers']/Seats['Seats']
Utilization = pd.DataFrame({'month': Passengers['month'], 'Utilization': Utilization})
Utilization['month'] = pd.Categorical(Utilization['month'], categories=month_order, ordered=True)
Utilization = Utilization.sort_values('month')

# Utilization of seats by Months
plt.figure(figsize=(10, 6))
plt.bar(Utilization['month'], Utilization['Utilization'], color='purple')
plt.xlabel('Month')
plt.ylabel('Average Utilization')
plt.show()

st.subheader("Airline A Seat Utilization by Months (1990-2008)")
st.pyplot(fig)
plt.clf()

APPF = Passengers['Passengers']/Flights['Flights']
APPF = pd.DataFrame({'month': Passengers['month'], 'APPF': APPF})
APPF['month'] = pd.Categorical(APPF['month'], categories=month_order, ordered=True)
APPF = APPF.sort_values('month')

# Average passenger per flight by Months
plt.figure(figsize=(10, 6))
plt.bar(APPF['month'], APPF['APPF'], color='purple')
plt.xlabel('Month')
plt.ylabel('Average Passenger Count')
plt.show()

st.subheader("Airline A Average Passengers per Flights (1990-2008)")
st.pyplot(fig)
plt.clf()
