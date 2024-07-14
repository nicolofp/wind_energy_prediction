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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

X_train = dt.loc[dt["Time"] < datetime(2021,1,1,0,0,0)].drop(labels=['Time','Power','tday_week',
                                                                 'tyear','tmonth','thour',
                                                                 'tday','date'], axis=1)
y_train = dt.loc[dt["Time"] < datetime(2021,1,1,0,0,0),'Power']
X_test = dt.loc[dt["Time"] >= datetime(2021,1,1,0,0,0)].drop(labels=['Time','Power','tday_week',
                                                                    'tyear','tmonth','thour',
                                                                    'tday','date'], axis=1)
y_test = dt.loc[dt["Time"] >= datetime(2021,1,1,0,0,0),'Power']


model = CatBoostRegressor(objective='Poisson')
#model = CatBoostRegressor(objective='RMSE')

catboost_param_dist = {
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.3),
    'iterations': randint(10, 1000),
    'l2_leaf_reg': randint(1, 10),
    'border_count': randint(1, 255),
    'bagging_temperature': uniform(0.0, 1.0),
    'random_strength': uniform(0.0, 1.0)
}

random_search_cb = RandomizedSearchCV(estimator=model,
                                      param_distributions=catboost_param_dist,
                                      cv=10,
                                      verbose=100,
                                      random_state=42)
# Fit the model
random_search_cb.fit(X_train, y_train)

# Evaluate and predict
predictions = random_search_cb.predict(X_test)
actual_counts = y_test.values.astype(int) 

r2_poisson = r2_score(actual_counts, predictions)
rmse_score_poisson_model = np.sqrt(mean_squared_error(actual_counts, predictions))
print('R2 score: {:.3f}\nRMSE score: {:.2f}'.format(r2_poisson, rmse_score_poisson_model))

dt_test["prediction"] = predictions
dt_day_test = pd.pivot_table(dt_test,
                             index=['date'],
                             aggfunc={'Power': np.mean, 'prediction': np.mean}).reset_index()
dt_day_test['date'] = pd.to_datetime(dt_day_test['date'], format="%Y-%m-%d")

plt.plot(dt_day_test['date'], dt_day_test['Power'], label = "Actual") 
plt.plot(dt_day_test['date'], dt_day_test['prediction'], label = "Predict") 
plt.legend() 
plt.show()