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


#Introduction to the Dashboard
st.write("Introduction to the Dashboard:")
st.write("Project Group One is acting as hypothetical consultants for an imaginary large American airline.")
st.write("The airline has posed a critical business question: Should they purchase more planes for their fleet?")
st.write("Our goal is to answer this using data visualization and analysis.")

#Title of the app
st.title("Airline A Dashboard")
st.write("PART ONE: What is the current status of the Client Airline’s Fleet?")

#Data reading

df = pd.read_csv("Airports_P 1.csv")
dt = pd.read_csv("Airports_T 1.csv")
dd = pd.read_csv("Airports_D.csv")

import streamlit.components.v1 as components
from streamlit.components.v1 import html

# Sidebar year slider
selected_year = st.sidebar.slider(
    "Select Year:",
    int(df["Year"].min()),
    int(df["Year"].max()),
    int(df["Year"].min()),
    step=4
)

# Filter by selected year
filtered_df = df[df["Year"] == selected_year]

# Extract unique airports
airports = filtered_df[['Origin_airport', 'Org_airport_lat', 'Org_airport_long', 'Origin_population']].drop_duplicates(subset=['Origin_airport'])
airports.columns = ['Airport', 'Latitude', 'Longitude', 'Population']
airports = airports.dropna()

# Create folium map centered on the US
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add airports as circle markers
for _, row in airports.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=math.sqrt(math.sqrt(float(row["Population"]))) / 5,
        popup=f"Airport: {row['Airport']}<br>Population: {row['Population']}",
        color="blue",
        fill=True,
    ).add_to(m)

# Display the map in Streamlit
st.subheader("Map of US airports")
map_html = m._repr_html_()
html(map_html, height=500, width=700)

st.write("Need for Airports:")
st.write("From 1990 onward, the number of airports increased, potentially signaling higher demand for air travel.")
st.write("However, this growth outpaces population increases, suggesting a shift in air travel infrastructure or strategy rather than demand alone.")
st.write("Learning:")
st.write("+  The number of airports is increasing")
st.write("+  Airport networks are becoming more complex")

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

# Display the plots in Streamlit
st.subheader("Airline A Monthly Flights with Rolling Average (3 months)")
st.plotly_chart(fig2)
plt.clf()

st.write("Changes in Flights:")
st.write("Flights steadily increased over time, followed by a decline in recent years.")
st.write("Our time series analysis also revealed consistent seasonal fluctuations, which complicates assumptions about constant demand growth.")
st.write("Learning:")
st.write("+  Flights increased over time until recently")
st.write("+  Seasonal fluctuations are consistent year-to-year")


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

st.write("The Average Flight Distance:")
st.write("Most flights are short-haul, peaking around 250 miles.")
st.write("While this doesn’t directly answer whether more planes are needed, it suggests a preference or need for smaller aircraft.")
st.write("Learning:")
st.write("+  Most flights cover short distances")

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

# Display the plot in Streamlit
st.subheader("Airline A Flights Forecast")
st.pyplot(plt)
plt.clf()

st.write("Future Flights:")
st.write("Predictive modeling shows that the total number of flights will likely remain steady.")
st.write("This raises questions about how flight frequency relates to demand, especially when considered alongside seat and passenger data.")
st.write("Learning:")
st.write("+  The number of future flights is expected to remain steady")

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

# Display the plots in Streamlit
st.subheader("Airline A Monthly Passengers with Rolling Average (3 months)")
st.plotly_chart(fig1)
plt.clf()

st.write("Increase in Passengers:")
st.write("While the number of flights has plateaued, passenger counts continue to grow.")
st.write("This suggests increasing demand for air travel despite fewer flights, emphasizing the importance of per-flight capacity.")
st.write("Learning:")
st.write("+  Passenger counts have steadily increased over time")

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
st.subheader("Airline A Passenger Forecast")
st.pyplot(plt)
plt.clf()

st.write("Change in Passengers:")
st.write("Passenger growth also shows a recent slowdown, similar to flight trends.")
st.write("However, this dip is less dramatic and suggests potential for renewed growth in the future.")
st.write("Learning:")
st.write("+  Indicators suggest continued growth in passenger volume")

st.write("PART TWO: Seat Utilization")

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

st.write("Continued Exploration on Passengers and Seats:")
st.write("By aggregating data monthly, we identified strong seasonality.")
st.write("While seats and flights scale together as expected, passengers do not always follow the same trend, especially in lower-demand months like February.")
st.write("This reveals months with high empty seat counts, a key inefficiency.")
st.write("Learning:")
st.write("+  Seasonal demand peaks in spring and summer")
st.write("+  Some months have excess seat capacity compared to passengers")

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
st.pyplot(plt)
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
st.pyplot(plt)
plt.clf()

st.write("Seat Utilization:")
st.write("Average passengers per flight range between 60 and 80.")
st.write("While we lack details on plane size distribution, this aligns with the predominance of short-distance flights and suggests an opportunity to better match plane size to seasonal demand.")
st.write("Learning:")
st.write("+  Seat utilization dips below optimal levels during parts of the year")

st.subheader("Final Recommendation")
st.write("Project Group One recommends not purchasing additional planes at this time. Instead, we advise the Client Airline to improve utilization of their existing fleet. Seasonal demand should guide scheduling and aircraft size. During low-demand months, consider reducing flight frequency or using smaller planes. If plane purchases are necessary in the future, prioritize smaller aircraft to align with short-haul trends and underutilization patterns.")


