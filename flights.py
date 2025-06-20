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
fig = px.line(
    monthly_passengers,
    x='Fly_date',
    y=['Passengers', 'Rolling_Avg'],
    title='Monthly Passengers with Rolling Average (3 months)',
    color_discrete_map={
        'Passengers': 'blue',
        'Rolling_Avg': 'green'
    }
)
fig.show(0)

# Display the plot in Streamlit
st.plotly_chart(fig)
