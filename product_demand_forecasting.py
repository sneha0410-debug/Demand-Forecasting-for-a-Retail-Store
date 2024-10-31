import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

data = pd.read_csv('sales_data.csv', parse_dates=['Date'], index_col='Date')
print(data.head())

data = data.ffill()

plt.figure(figsize=(10, 6))
plt.plot(data, label='Historical Sales')
plt.title('Product Demand Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

if len(data) < 12:
    print("Warning: Data may be insufficient for seasonality modeling. Consider adding more data points.")

train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

model = SARIMAX(train_data, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
sarima_model = model.fit(disp=False)
print(sarima_model.summary())

forecast_steps = len(test_data) if len(test_data) > 0 else 1 
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = test_data.index if len(test_data) > 0 else pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='MS')[1:]

plt.figure(figsize=(10, 6))
plt.plot(data, label='Historical Sales')
plt.plot(forecast_index, forecast.predicted_mean, color='red', label='Forecast')

conf_int = forecast.conf_int() if 'lower Sales' in forecast.conf_int().columns else None
if conf_int is not None:
    plt.fill_between(forecast_index,
                     conf_int['lower Sales'],
                     conf_int['upper Sales'], color='pink', alpha=0.3)

plt.title('Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
