import pandas as pd
import scipy as sp
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

dt = pd.read_csv("C:/Users/nicol/Documents/wind_production/Location1.csv")
dt.drop("Time", axis=1).corr().round(2)

# Convert data into datetime format
dt["Time"] = pd.to_datetime(dt["Time"])
dt["tday_week"] = dt["Time"].dt.day_of_week
dt["tyear"] = dt["Time"].dt.year
dt["tmonth"] = dt["Time"].dt.month
dt["thour"] = dt["Time"].dt.hour
dt["tday"] = dt["Time"].dt.day 
dt["date"] = dt["Time"].dt.date

# Convert wind direction
dt['winddirection_10m_cos']  = np.cos(dt['winddirection_10m'] * np.pi/180)
dt['winddirection_10m_sin']  = np.sin(dt['winddirection_10m'] * np.pi/180)
dt['winddirection_100m_cos'] = np.cos(dt['winddirection_100m'] * np.pi/180)
dt['winddirection_100m_sin'] = np.sin(dt['winddirection_100m'] * np.pi/180)

# dt.loc[dt['winddirection_10m'].isin([90,270]),['winddirection_10m','winddirection_10m_sin','winddirection_10m_cos']]
dt["Power"] = 100 * dt["Power"].round(2)
plt.hist(dt["Power"], bins=50)
plt.show()

# Cyclic encoder function
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def cyclic_encoder(df, names_columns, column_max):
    for i, j in zip(names_columns, column_max):
        df[i + '_sin'] = np.sin(2 * np.pi * df[i]/j)
        df[i + '_cos'] = np.cos(2 * np.pi * df[i]/j)
    return df

names_columns = dt.columns[11:15]
column_max = [2021, 12, 23, 31]
cyclic_encoder(df = dt, names_columns = names_columns, column_max = column_max)    
  

plt.scatter(dt["windspeed_100m"], dt["Power"])
plt.show()

plt.plot(dt.loc[dt["Time"] > datetime(2020,12,31,23,0,0),"Time"], 
         dt.loc[dt["Time"] > datetime(2020,12,31,23,0,0),"Power"]) 
plt.show()

dt_day = pd.pivot_table(dt,
               index=['date'],
               aggfunc={'temperature_2m': np.mean, 'relativehumidity_2m': np.mean, 
                        'dewpoint_2m': np.mean, 'windspeed_10m': np.mean, 
                        'windspeed_100m': np.mean, 'windgusts_10m': np.mean, 
                        'Power': np.mean}).reset_index()
dt_day['date'] = pd.to_datetime(dt_day['date'], format="%Y-%m-%d")

plt.scatter(dt_day["windspeed_10m"], dt_day["Power"])
plt.show()

plt.scatter(dt["windspeed_10m"], dt["Power"])
plt.show()

plt.scatter(dt.loc[(dt["windspeed_10m"] < 5) & (dt["Power"] >= 0),["windspeed_10m"]],
            dt.loc[(dt["windspeed_10m"] < 5) & (dt["Power"] >= 0),["Power"]])
plt.show()

plt.hist(dt["Power"])
plt.show()

dt.describe()
dt.dropna()
dt.drop_duplicates()

plt.plot(dt_day.loc[dt_day["date"] > datetime(2020,12,31),"date"], 
         dt_day.loc[dt_day["date"] > datetime(2020,12,31),"Power"]) 
plt.show()

# Machine learning

from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import r2_score, mean_squared_error

dt_train = dt[dt["Time"] < datetime(2019,7,1,0,0,0)]
dt_val = dt[(dt["Time"] >= datetime(2019,7,1,0,0,0)) & (dt["Time"] <= datetime(2020,12,31,23,0,0))]
dt_test = dt[dt["Time"] > datetime(2020,12,31,23,0,0)]

train_pool = Pool(dt_train.drop(labels=['Time','Power','tday_week','tyear','tmonth', 
                                        'thour', 'tday','date'], axis=1),
                  label=dt_train.Power.values.astype(int))

val_pool = Pool(dt_val.drop(labels=['Time','Power','tday_week', 'tyear',
                             'tmonth', 'thour', 'tday', 'date'], axis=1),
                label=dt_val.Power.values.astype(int))

test_pool = Pool(dt_test.drop(labels=['Time','Power','tday_week', 'tyear',
                               'tmonth', 'thour', 'tday', 'date'], axis=1),
                label=dt_test.Power.values.astype(int))

model = CatBoostRegressor(objective='Poisson')
#model = CatBoostRegressor(objective='RMSE')

model.fit(train_pool, plot=True, eval_set=val_pool, verbose=500)

actual_counts = dt_test.Power.values.astype(int) 
predicted_counts_poisson = model.predict(test_pool) 
r2_poisson = r2_score(actual_counts, predicted_counts_poisson)
rmse_score_poisson_model = np.sqrt(mean_squared_error(actual_counts, predicted_counts_poisson))
print('R2 score: {:.3f}\nRMSE score: {:.2f}'.format(r2_poisson, rmse_score_poisson_model))

dt_test["prediction"] = predicted_counts_poisson
dt_day_test = pd.pivot_table(dt_test,
                             index=['date'],
                             aggfunc={'Power': np.mean, 'prediction': np.mean}).reset_index()
dt_day_test['date'] = pd.to_datetime(dt_day_test['date'], format="%Y-%m-%d")

plt.plot(dt_day_test['date'], dt_day_test['Power'], label = "Actual") 
plt.plot(dt_day_test['date'], dt_day_test['prediction'], label = "Predict") 
plt.legend() 
plt.show()