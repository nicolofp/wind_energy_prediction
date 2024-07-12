import pandas as pd
import scipy as sp
import numpy as np
from datetime import datetime

dt = pd.read_csv("C:/Users/nicol/Documents/wind_production/Location1.csv")
dt.drop("Time", axis=1).corr().round(2)

# Convert data into datetime format
dt["Time"] = pd.to_datetime(dt["Time"])
dt["tday_week"] = dt["Time"].dt.day_of_week
dt["tyear"] = dt["Time"].dt.year
dt["tmonth"] = dt["Time"].dt.month
dt["thour"] = dt["Time"].dt.hour
dt["tday"] = dt["Time"].dt.day 

# Cyclic encoder










