# %% [markdown]
# # Predictions

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
data1 = pd.read_csv("Datasets1/warangal_hw.csv",error_bad_lines=False)
data2 = pd.read_csv("Datasets1/nizamabad_hw.csv",error_bad_lines=False)
data3 = pd.read_csv("Datasets1/khammam_hw.csv",error_bad_lines=False)
data4 = pd.read_csv("Datasets1/karimnagar_hw.csv",error_bad_lines=False)
data5 = pd.read_csv("Datasets1/adilabad_hw.csv",error_bad_lines=False)

# %%
l=[data1,data2,data3,data4,data5]

# %%
dates = pd.date_range("01/01/1982","31/12/2022",inclusive="both")
dates

# %%
for i in l:
    i["dates"] = dates
    i.set_index(dates,inplace=True)

# %%
for i in l:
    i.drop(["YEAR","MO","DY"],axis=1,inplace=True)

# %% [markdown]
# ## Warangal

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp1 = seasonal_decompose(l[0]["T2M"],model="additive")
seasonality1 = decomp1.seasonal
trend1 = decomp1.trend
error1 = decomp1.resid

# %%
l[0]["T2M"].iloc[14931:14934] =[20.08,20.1,20] #l[0]["T2M"].iloc[14928:14936].mean()
l[0]["QV2M"].iloc[14931:14934] =[11.6,12.45,12.7]
l[0]["WS10M"].iloc[14931:14934] =[2.82,2.54,2.35]

# %%
weekly = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[0])):
        temp = l[0]["T2M"].iloc[i:].sum()
        weekly.append(temp)
        break
    temp = l[0]["T2M"].iloc[i:i+7].sum()
    weekly.append(temp)
    i+=7

for i in range(len(weekly)):
    weekly[i]/=7

weekly = weekly[:-1]

# %%
weekly = pd.DataFrame(weekly)
weekly["dates"] = dates
weekly.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
weekly_decomp=seasonal_decompose(weekly)
weekly_seasonal = weekly_decomp.seasonal
weekly_trend = weekly_decomp.trend
weekly_res = weekly_decomp.resid

# %%
weekly_hum = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[0])):
        temp = l[0]["QV2M"].iloc[i:].sum()
        weekly_hum.append(temp)
        break
    temp = l[0]["QV2M"].iloc[i:i+7].sum()
    weekly_hum.append(temp)
    i+=7

for i in range(len(weekly_hum)):
    weekly_hum[i] /= 7
weekly_hum = weekly_hum[:-1]

# %%
weekly_hum = pd.DataFrame(weekly_hum)
weekly_hum["dates"] = dates
weekly_hum.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
hum_decomp=seasonal_decompose(weekly_hum)
hum_seasonal = hum_decomp.seasonal
hum_trend = hum_decomp.trend
hum_res = hum_decomp.resid

# %%
weekly_ws = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[0])):
        temp = l[0]["WS10M"].iloc[i:].sum()
        weekly_ws.append(temp)
        break
    temp = l[0]["WS10M"].iloc[i:i+7].sum()
    weekly_ws.append(temp)
    i+=7

for i in range(len(weekly_ws)):
    weekly_ws[i] /= 7
weekly_ws = weekly_ws[:-1]

# %%
weekly_ws = pd.DataFrame(weekly_ws)
weekly_ws["dates"] = dates
weekly_ws.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
ws_decomp=seasonal_decompose(weekly_ws)
ws_seasonal = ws_decomp.seasonal
ws_trend = ws_decomp.trend
ws_res = ws_decomp.resid

# %% [markdown]
# ### Trend 

# %%
train = weekly_trend[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_obj1 = ARIMA(train,order = (2,1,0),seasonal_order = (1,1,1,52))
model1 = arima_obj1.fit()

# %%
fore1 = model1.forecast(52+26)
final_fore1 = pd.DataFrame(fore1)
plt.plot(final_fore1)

# %%
final_fore1 = final_fore1.iloc[26:,:]

# %%
final_fore1

# %% [markdown]
# ### seasonal

# %%
train1 = weekly_seasonal[-366:]

# %%
from statsmodels.tsa.arima.model import ARIMA
season_obj = ARIMA(train1,order = (1,0,1),seasonal_order = (4,0,4,26))
season_model = season_obj.fit()

# %%
final_fore2 = season_model.forecast(52)
plt.plot(season_model.forecast(52))

# %% [markdown]
# ### Residual 

# %%
train2 = weekly_res[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
res_obj = ARIMA(train2,order = (0,1,0),seasonal_order = (1,2,1,52))
res_model = res_obj.fit()

# %%
final_fore3 = res_model.forecast(52+26)
final_fore3 = final_fore3[26:]
plt.plot(final_fore3)

# %% [markdown]
# ### LSTM

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.layers import Dropout

# %%
tf.random.set_seed(7)

# %%
seq = Sequential()
seq.add(Dropout(0.01))
seq.add(LSTM(365, input_shape = (1,3))) # bias_regularizer=L1L2(0.1,0.24)))
seq.add(Dense(1))
seq.compile(optimizer = 'Adam', loss = 'mae', metrics = ['MAPE'])

# %%
x_final = np.concatenate((weekly_seasonal[26:-26].values.reshape(-1,1),weekly_trend[26:-26].values.reshape(-1,1),weekly_res[26:-26].values.reshape(-1,1)),axis = 1)
y_final = weekly[26:-26].values.reshape(-1,1)

# %%
x_final = np.reshape(x_final,(x_final.shape[0],1,x_final.shape[1]))
y_final = np.reshape(y_final,(y_final.shape[0],1,y_final.shape[1]))

# %%
x_final_train = x_final[0:(int(0.9 * len(x_final)))]
x_final_test = x_final[(int(0.9 * len(x_final))):]
print(x_final_train.shape)
print(x_final_test.shape)

# %%
y_final_train = y_final[0:(int(0.9 * len(y_final)))]
y_final_test = y_final[(int(0.9 * len(y_final))):]
print(y_final_train.shape)
print(y_final_test.shape)

# %%
y_final.shape
x_final.shape

# %%
seq.fit(x_final_train, y_final_train, epochs=50, batch_size = 64, validation_data = (x_final_test, y_final_test))

# %%
actual = np.concatenate((final_fore1.values.reshape(-1,1),final_fore2.values.reshape(-1,1),final_fore3.values.reshape(-1,1)),axis = 1)
actual = np.reshape(actual, (actual.shape[0], 1, actual.shape[1]))
pred = seq.predict(actual)
plt.plot(pred, c = 'blue')

# %%
warangal = []
for i in pred:
    warangal += [i[0]]
print(warangal)

# %% [markdown]
# ## Nizamabad 

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp1 = seasonal_decompose(l[1]["T2M"],model="additive")
seasonality1 = decomp1.seasonal
trend1 = decomp1.trend
error1 = decomp1.resid

# %%
l[1]["T2M"].iloc[14931:14934] =[19.52,19.8,20]
l[1]["QV2M"].iloc[14931:14934] =[11.4,11.97,12.3]
l[1]["WS10M"].iloc[14931:14934] =[2.29,2.52,2.81]

# %%
weekly = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[1])):
        temp = l[1]["T2M"].iloc[i:].sum()
        weekly.append(temp)
        break
    temp = l[1]["T2M"].iloc[i:i+7].sum()
    weekly.append(temp)
    i+=7

for i in range(len(weekly)):
    weekly[i]/=7

weekly = weekly[:-1]

# %%
weekly = pd.DataFrame(weekly)
weekly["dates"] = dates
weekly.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
weekly_decomp=seasonal_decompose(weekly)
weekly_seasonal = weekly_decomp.seasonal
weekly_trend = weekly_decomp.trend
weekly_res = weekly_decomp.resid

# %%
weekly_hum = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[1])):
        temp = l[1]["QV2M"].iloc[i:].sum()
        weekly_hum.append(temp)
        break
    temp = l[1]["QV2M"].iloc[i:i+7].sum()
    weekly_hum.append(temp)
    i+=7

for i in range(len(weekly_hum)):
    weekly_hum[i] /= 7
weekly_hum = weekly_hum[:-1]

# %%
weekly_hum = pd.DataFrame(weekly_hum)
weekly_hum["dates"] = dates
weekly_hum.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
hum_decomp=seasonal_decompose(weekly_hum)
hum_seasonal = hum_decomp.seasonal
hum_trend = hum_decomp.trend
hum_res = hum_decomp.resid

# %%
weekly_ws = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[1])):
        temp = l[1]["WS10M"].iloc[i:].sum()
        weekly_ws.append(temp)
        break
    temp = l[1]["WS10M"].iloc[i:i+7].sum()
    weekly_ws.append(temp)
    i+=7

for i in range(len(weekly_ws)):
    weekly_ws[i] /= 7
weekly_ws = weekly_ws[:-1]

# %%
weekly_ws = pd.DataFrame(weekly_ws)
weekly_ws["dates"] = dates
weekly_ws.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
ws_decomp=seasonal_decompose(weekly_ws)
ws_seasonal = ws_decomp.seasonal
ws_trend = ws_decomp.trend
ws_res = ws_decomp.resid

# %% [markdown]
# ### Trend 

# %%
train = weekly_trend[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_obj1 = ARIMA(train,order = (2,1,0),seasonal_order = (1,1,1,52))
model1 = arima_obj1.fit()

# %%
fore1 = model1.forecast(52+26)
final_fore1 = pd.DataFrame(fore1)
plt.plot(final_fore1)

# %%
final_fore1 = final_fore1.iloc[26:,:]

# %%
final_fore1

# %% [markdown]
# ### seasonal

# %%
train1 = weekly_seasonal[-366:]

# %%
from statsmodels.tsa.arima.model import ARIMA
season_obj = ARIMA(train1,order = (1,0,1),seasonal_order = (4,0,4,26))
season_model = season_obj.fit()

# %%
final_fore2 = season_model.forecast(52)
plt.plot(season_model.forecast(52))

# %% [markdown]
# ### Residual 

# %%
train2 = weekly_res[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
res_obj = ARIMA(train2,order = (0,1,0),seasonal_order = (1,2,1,52))
res_model = res_obj.fit()

# %%
final_fore3 = res_model.forecast(52+26)
final_fore3 = final_fore3[26:]
plt.plot(final_fore3)

# %% [markdown]
# ### LSTM

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.layers import Dropout

# %%
tf.random.set_seed(7)

# %%
seq = Sequential()
seq.add(Dropout(0.01))
seq.add(LSTM(365, input_shape = (1,3))) # bias_regularizer=L1L2(0.1,0.24)))
seq.add(Dense(1))
seq.compile(optimizer = 'Adam', loss = 'mae', metrics = ['MAPE'])

# %%
x_final = np.concatenate((weekly_seasonal[26:-26].values.reshape(-1,1),weekly_trend[26:-26].values.reshape(-1,1),weekly_res[26:-26].values.reshape(-1,1)),axis = 1)
y_final = weekly[26:-26].values.reshape(-1,1)

# %%
x_final = np.reshape(x_final,(x_final.shape[0],1,x_final.shape[1]))
y_final = np.reshape(y_final,(y_final.shape[0],1,y_final.shape[1]))

# %%
x_final_train = x_final[0:(int(0.9 * len(x_final)))]
x_final_test = x_final[(int(0.9 * len(x_final))):]
print(x_final_train.shape)
print(x_final_test.shape)

# %%
y_final_train = y_final[0:(int(0.9 * len(y_final)))]
y_final_test = y_final[(int(0.9 * len(y_final))):]
print(y_final_train.shape)
print(y_final_test.shape)

# %%
y_final.shape
x_final.shape

# %%
seq.fit(x_final_train, y_final_train, epochs=50, batch_size = 64, validation_data = (x_final_test, y_final_test))

# %%
actual = np.concatenate((final_fore1.values.reshape(-1,1),final_fore2.values.reshape(-1,1),final_fore3.values.reshape(-1,1)),axis = 1)
actual = np.reshape(actual, (actual.shape[0], 1, actual.shape[1]))
pred = seq.predict(actual)
plt.plot(pred, c = 'blue')

# %%
nizamabad = []
for i in pred:
    nizamabad += [i[0]]
print(nizamabad)

# %% [markdown]
# ## Khammam

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp1 = seasonal_decompose(l[2]["T2M"],model="additive")
seasonality1 = decomp1.seasonal
trend1 = decomp1.trend
error1 = decomp1.resid

# %%
l[2]["WS10M"].iloc[14928:14936]

# %%
l[2]["T2M"].iloc[14931:14934] =[21.15,21.55,22] 
l[2]["QV2M"].iloc[14931:14934] =[13.24,12.85,13.83]
l[2]["WS10M"].iloc[14931:14934] =[2.47,2.88,2.97]

# %%
weekly = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[2])):
        temp = l[2]["T2M"].iloc[i:].sum()
        weekly.append(temp)
        break
    temp = l[2]["T2M"].iloc[i:i+7].sum()
    weekly.append(temp)
    i+=7

for i in range(len(weekly)):
    weekly[i]/=7

weekly = weekly[:-1]

# %%
weekly = pd.DataFrame(weekly)
weekly["dates"] = dates
weekly.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
weekly_decomp=seasonal_decompose(weekly)
weekly_seasonal = weekly_decomp.seasonal
weekly_trend = weekly_decomp.trend
weekly_res = weekly_decomp.resid

# %%
weekly_hum = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[2])):
        temp = l[2]["QV2M"].iloc[i:].sum()
        weekly_hum.append(temp)
        break
    temp = l[2]["QV2M"].iloc[i:i+7].sum()
    weekly_hum.append(temp)
    i+=7

for i in range(len(weekly_hum)):
    weekly_hum[i] /= 7
weekly_hum = weekly_hum[:-1]

# %%
weekly_hum = pd.DataFrame(weekly_hum)
weekly_hum["dates"] = dates
weekly_hum.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
hum_decomp=seasonal_decompose(weekly_hum)
hum_seasonal = hum_decomp.seasonal
hum_trend = hum_decomp.trend
hum_res = hum_decomp.resid

# %%
weekly_ws = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[2])):
        temp = l[2]["WS10M"].iloc[i:].sum()
        weekly_ws.append(temp)
        break
    temp = l[2]["WS10M"].iloc[i:i+7].sum()
    weekly_ws.append(temp)
    i+=7

for i in range(len(weekly_ws)):
    weekly_ws[i] /= 7
weekly_ws = weekly_ws[:-1]

# %%
weekly_ws = pd.DataFrame(weekly_ws)
weekly_ws["dates"] = dates
weekly_ws.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
ws_decomp=seasonal_decompose(weekly_ws)
ws_seasonal = ws_decomp.seasonal
ws_trend = ws_decomp.trend
ws_res = ws_decomp.resid

# %% [markdown]
# ### Trend 

# %%
train = weekly_trend[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_obj1 = ARIMA(train,order = (2,1,0),seasonal_order = (1,1,1,52))
model1 = arima_obj1.fit()

# %%
fore1 = model1.forecast(52+26)
final_fore1 = pd.DataFrame(fore1)
plt.plot(final_fore1)

# %%
final_fore1 = final_fore1.iloc[26:,:]

# %%
final_fore1

# %% [markdown]
# ### seasonal

# %%
train1 = weekly_seasonal[-366:]

# %%
from statsmodels.tsa.arima.model import ARIMA
season_obj = ARIMA(train1,order = (1,0,1),seasonal_order = (4,0,4,26))
season_model = season_obj.fit()

# %%
final_fore2 = season_model.forecast(52)
plt.plot(season_model.forecast(52))

# %% [markdown]
# ### Residual 

# %%
train2 = weekly_res[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
res_obj = ARIMA(train2,order = (0,1,0),seasonal_order = (1,2,1,52))
res_model = res_obj.fit()

# %%
final_fore3 = res_model.forecast(52+26)
final_fore3 = final_fore3[26:]
plt.plot(final_fore3)

# %% [markdown]
# ### LSTM

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.layers import Dropout

# %%
tf.random.set_seed(7)

# %%
seq = Sequential()
seq.add(Dropout(0.01))
seq.add(LSTM(365, input_shape = (1,3))) # bias_regularizer=L1L2(0.1,0.24)))
seq.add(Dense(1))
seq.compile(optimizer = 'Adam', loss = 'mae', metrics = ['MAPE'])

# %%
x_final = np.concatenate((weekly_seasonal[26:-26].values.reshape(-1,1),weekly_trend[26:-26].values.reshape(-1,1),weekly_res[26:-26].values.reshape(-1,1)),axis = 1)
y_final = weekly[26:-26].values.reshape(-1,1)

# %%
x_final = np.reshape(x_final,(x_final.shape[0],1,x_final.shape[1]))
y_final = np.reshape(y_final,(y_final.shape[0],1,y_final.shape[1]))

# %%
x_final_train = x_final[0:(int(0.9 * len(x_final)))]
x_final_test = x_final[(int(0.9 * len(x_final))):]
print(x_final_train.shape)
print(x_final_test.shape)

# %%
y_final_train = y_final[0:(int(0.9 * len(y_final)))]
y_final_test = y_final[(int(0.9 * len(y_final))):]
print(y_final_train.shape)
print(y_final_test.shape)

# %%
y_final.shape
x_final.shape

# %%
seq.fit(x_final_train, y_final_train, epochs=50, batch_size = 64, validation_data = (x_final_test, y_final_test))

# %%
actual = np.concatenate((final_fore1.values.reshape(-1,1),final_fore2.values.reshape(-1,1),final_fore3.values.reshape(-1,1)),axis = 1)
actual = np.reshape(actual, (actual.shape[0], 1, actual.shape[1]))
pred = seq.predict(actual)
plt.plot(pred, c = 'blue')

# %%
khammam = []
for i in pred:
    khammam += [i[0]]
print(khammam)

# %% [markdown]
# ## Karimnagar

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp1 = seasonal_decompose(l[3]["T2M"],model="additive")
seasonality1 = decomp1.seasonal
trend1 = decomp1.trend
error1 = decomp1.resid

# %%
l[3]["T2M"].iloc[14931:14934] =[20.12,20.55,20.9] 
l[3]["QV2M"].iloc[14931:14934] =[12.16,12.7,13]
l[3]["WS10M"].iloc[14931:14934] =[2.28,2.54,2.89]

# %%
weekly = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[3])):
        temp = l[3]["T2M"].iloc[i:].sum()
        weekly.append(temp)
        break
    temp = l[3]["T2M"].iloc[i:i+7].sum()
    weekly.append(temp)
    i+=7

for i in range(len(weekly)):
    weekly[i]/=7

weekly = weekly[:-1]

# %%
weekly = pd.DataFrame(weekly)
weekly["dates"] = dates
weekly.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
weekly_decomp=seasonal_decompose(weekly)
weekly_seasonal = weekly_decomp.seasonal
weekly_trend = weekly_decomp.trend
weekly_res = weekly_decomp.resid

# %%
weekly_hum = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[3])):
        temp = l[3]["QV2M"].iloc[i:].sum()
        weekly_hum.append(temp)
        break
    temp = l[3]["QV2M"].iloc[i:i+7].sum()
    weekly_hum.append(temp)
    i+=7

for i in range(len(weekly_hum)):
    weekly_hum[i] /= 7
weekly_hum = weekly_hum[:-1]

# %%
weekly_hum = pd.DataFrame(weekly_hum)
weekly_hum["dates"] = dates
weekly_hum.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
hum_decomp=seasonal_decompose(weekly_hum)
hum_seasonal = hum_decomp.seasonal
hum_trend = hum_decomp.trend
hum_res = hum_decomp.resid

# %%
weekly_ws = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[3])):
        temp = l[3]["WS10M"].iloc[i:].sum()
        weekly_ws.append(temp)
        break
    temp = l[3]["WS10M"].iloc[i:i+7].sum()
    weekly_ws.append(temp)
    i+=7

for i in range(len(weekly_ws)):
    weekly_ws[i] /= 7
weekly_ws = weekly_ws[:-1]

# %%
weekly_ws = pd.DataFrame(weekly_ws)
weekly_ws["dates"] = dates
weekly_ws.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
ws_decomp=seasonal_decompose(weekly_ws)
ws_seasonal = ws_decomp.seasonal
ws_trend = ws_decomp.trend
ws_res = ws_decomp.resid

# %% [markdown]
# ### Trend 

# %%
train = weekly_trend[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_obj1 = ARIMA(train,order = (2,1,0),seasonal_order = (1,1,1,52))
model1 = arima_obj1.fit()

# %%
fore1 = model1.forecast(52+26)
final_fore1 = pd.DataFrame(fore1)
plt.plot(final_fore1)

# %%
final_fore1 = final_fore1.iloc[26:,:]

# %%
final_fore1

# %% [markdown]
# ### seasonal

# %%
train1 = weekly_seasonal[-366:]

# %%
from statsmodels.tsa.arima.model import ARIMA
season_obj = ARIMA(train1,order = (1,0,1),seasonal_order = (4,0,4,26))
season_model = season_obj.fit()

# %%
final_fore2 = season_model.forecast(52)
plt.plot(season_model.forecast(52))

# %% [markdown]
# ### Residual 

# %%
train2 = weekly_res[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
res_obj = ARIMA(train2,order = (0,1,0),seasonal_order = (1,2,1,52))
res_model = res_obj.fit()

# %%
final_fore3 = res_model.forecast(52+26)
final_fore3 = final_fore3[26:]
plt.plot(final_fore3)

# %% [markdown]
# ### LSTM

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.layers import Dropout

# %%
tf.random.set_seed(7)

# %%
seq = Sequential()
seq.add(Dropout(0.01))
seq.add(LSTM(365, input_shape = (1,3))) # bias_regularizer=L1L2(0.1,0.24)))
seq.add(Dense(1))
seq.compile(optimizer = 'Adam', loss = 'mae', metrics = ['MAPE'])

# %%
x_final = np.concatenate((weekly_seasonal[26:-26].values.reshape(-1,1),weekly_trend[26:-26].values.reshape(-1,1),weekly_res[26:-26].values.reshape(-1,1)),axis = 1)
y_final = weekly[26:-26].values.reshape(-1,1)

# %%
x_final = np.reshape(x_final,(x_final.shape[0],1,x_final.shape[1]))
y_final = np.reshape(y_final,(y_final.shape[0],1,y_final.shape[1]))

# %%
x_final_train = x_final[0:(int(0.9 * len(x_final)))]
x_final_test = x_final[(int(0.9 * len(x_final))):]
print(x_final_train.shape)
print(x_final_test.shape)

# %%
y_final_train = y_final[0:(int(0.9 * len(y_final)))]
y_final_test = y_final[(int(0.9 * len(y_final))):]
print(y_final_train.shape)
print(y_final_test.shape)

# %%
y_final.shape
x_final.shape

# %%
seq.fit(x_final_train, y_final_train, epochs=50, batch_size = 64, validation_data = (x_final_test, y_final_test))

# %%
actual = np.concatenate((final_fore1.values.reshape(-1,1),final_fore2.values.reshape(-1,1),final_fore3.values.reshape(-1,1)),axis = 1)
actual = np.reshape(actual, (actual.shape[0], 1, actual.shape[1]))
pred = seq.predict(actual)
plt.plot(pred, c = 'blue')

# %%
karimnagar = []
for i in pred:
    karimnagar += [i[0]]
print(karimnagar)

# %% [markdown]
# ## Adilabad

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp1 = seasonal_decompose(l[4]["T2M"],model="additive")
seasonality1 = decomp1.seasonal
trend1 = decomp1.trend
error1 = decomp1.resid

# %%
l[4]["T2M"].iloc[14931:14934] =[19.14,20,19.73] 
l[4]["QV2M"].iloc[14931:14934] =[10.76,11.3,11.59]
l[4]["WS10M"].iloc[14931:14934] =[2.19,2.7,3]

# %%
weekly = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[4])):
        temp = l[4]["T2M"].iloc[i:].sum()
        weekly.append(temp)
        break
    temp = l[4]["T2M"].iloc[i:i+7].sum()
    weekly.append(temp)
    i+=7

for i in range(len(weekly)):
    weekly[i]/=7

weekly = weekly[:-1]

# %%
weekly = pd.DataFrame(weekly)
weekly["dates"] = dates
weekly.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
weekly_decomp=seasonal_decompose(weekly)
weekly_seasonal = weekly_decomp.seasonal
weekly_trend = weekly_decomp.trend
weekly_res = weekly_decomp.resid

# %%
weekly_hum = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[4])):
        temp = l[4]["QV2M"].iloc[i:].sum()
        weekly_hum.append(temp)
        break
    temp = l[4]["QV2M"].iloc[i:i+7].sum()
    weekly_hum.append(temp)
    i+=7

for i in range(len(weekly_hum)):
    weekly_hum[i] /= 7
weekly_hum = weekly_hum[:-1]

# %%
weekly_hum = pd.DataFrame(weekly_hum)
weekly_hum["dates"] = dates
weekly_hum.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
hum_decomp=seasonal_decompose(weekly_hum)
hum_seasonal = hum_decomp.seasonal
hum_trend = hum_decomp.trend
hum_res = hum_decomp.resid

# %%
weekly_ws = []
dates = pd.date_range("01/01/1982","31/12/2022",freq = "W-THU",inclusive="both")
i=0
while(True):
    if(i+7>len(l[4])):
        temp = l[4]["WS10M"].iloc[i:].sum()
        weekly_ws.append(temp)
        break
    temp = l[4]["WS10M"].iloc[i:i+7].sum()
    weekly_ws.append(temp)
    i+=7

for i in range(len(weekly_ws)):
    weekly_ws[i] /= 7
weekly_ws = weekly_ws[:-1]

# %%
weekly_ws = pd.DataFrame(weekly_ws)
weekly_ws["dates"] = dates
weekly_ws.set_index("dates",inplace=True)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
ws_decomp=seasonal_decompose(weekly_ws)
ws_seasonal = ws_decomp.seasonal
ws_trend = ws_decomp.trend
ws_res = ws_decomp.resid

# %% [markdown]
# ### Trend 

# %%
train = weekly_trend[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_obj1 = ARIMA(train,order = (2,1,0),seasonal_order = (1,1,1,52))
model1 = arima_obj1.fit()

# %%
fore1 = model1.forecast(52+26)
final_fore1 = pd.DataFrame(fore1)
plt.plot(final_fore1)

# %%
final_fore1 = final_fore1.iloc[26:,:]

# %%
final_fore1

# %% [markdown]
# ### seasonal

# %%
train1 = weekly_seasonal[-366:]

# %%
from statsmodels.tsa.arima.model import ARIMA
season_obj = ARIMA(train1,order = (1,0,1),seasonal_order = (4,0,4,26))
season_model = season_obj.fit()

# %%
final_fore2 = season_model.forecast(52)
plt.plot(season_model.forecast(52))

# %% [markdown]
# ### Residual 

# %%
train2 = weekly_res[-366:-26]

# %%
from statsmodels.tsa.arima.model import ARIMA
res_obj = ARIMA(train2,order = (0,1,0),seasonal_order = (1,2,1,52))
res_model = res_obj.fit()

# %%
final_fore3 = res_model.forecast(52+26)
final_fore3 = final_fore3[26:]
plt.plot(final_fore3)

# %% [markdown]
# ### LSTM

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.layers import Dropout

# %%
tf.random.set_seed(7)

# %%
seq = Sequential()
seq.add(Dropout(0.01))
seq.add(LSTM(365, input_shape = (1,3))) # bias_regularizer=L1L2(0.1,0.24)))
seq.add(Dense(1))
seq.compile(optimizer = 'Adam', loss = 'mae', metrics = ['MAPE'])

# %%
x_final = np.concatenate((weekly_seasonal[26:-26].values.reshape(-1,1),weekly_trend[26:-26].values.reshape(-1,1),weekly_res[26:-26].values.reshape(-1,1)),axis = 1)
y_final = weekly[26:-26].values.reshape(-1,1)

# %%
x_final = np.reshape(x_final,(x_final.shape[0],1,x_final.shape[1]))
y_final = np.reshape(y_final,(y_final.shape[0],1,y_final.shape[1]))

# %%
x_final_train = x_final[0:(int(0.9 * len(x_final)))]
x_final_test = x_final[(int(0.9 * len(x_final))):]
print(x_final_train.shape)
print(x_final_test.shape)

# %%
y_final_train = y_final[0:(int(0.9 * len(y_final)))]
y_final_test = y_final[(int(0.9 * len(y_final))):]
print(y_final_train.shape)
print(y_final_test.shape)

# %%
y_final.shape
x_final.shape

# %%
seq.fit(x_final_train, y_final_train, epochs=50, batch_size = 64, validation_data = (x_final_test, y_final_test))

# %%
actual = np.concatenate((final_fore1.values.reshape(-1,1),final_fore2.values.reshape(-1,1),final_fore3.values.reshape(-1,1)),axis = 1)
actual = np.reshape(actual, (actual.shape[0], 1, actual.shape[1]))
pred = seq.predict(actual)
plt.plot(pred, c = 'blue')

# %%
adilabad = []
for i in pred:
    adilabad += [i[0]]
print(adilabad)


