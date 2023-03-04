# %% [markdown]
# # Importing neseccary libraries for data

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Loading Datasets for AQI

# %%
aqi_data_6 = pd.read_excel("Datasets/MonthlyAQI2016.xlsx")
aqi_data_7 = pd.read_excel("Datasets/AQI_2017.xlsx")
aqi_data_8 = pd.read_excel("Datasets/Monthly_AQI_Jan_-_Dec_2018.xlsx")
aqi_data_9 = pd.read_excel("Datasets/MonthlyAQI2019.xlsx")
aqi_data_20 = pd.read_excel("Datasets/MonthlyAQI2020.xlsx")
aqi_data_21 = pd.read_excel("Datasets/MonthlyAQI2021.xlsx")
aqi_data_22 = pd.read_excel("Datasets/MonthlyAQI2022.xlsx")

# %%
l_aqi = [aqi_data_6.iloc[:,:-1],aqi_data_7.iloc[:,:-1],aqi_data_8,aqi_data_9,aqi_data_20,aqi_data_21,aqi_data_22]

# %% [markdown]
# # Cleaning the data and putting into perspective

# %% [markdown]
# ## L2 of the form nizamabad, adilabad, warangal, karimnagar,khammam

# %%
l2 = []
for i in range(0,5):
    aqi = pd.DataFrame()
    for j in l_aqi:
        aqi = pd.concat([aqi,j.iloc[i,:][1:]],ignore_index=True)
    aqi[aqi[0] == '-'] = np.nan
    l2.append(aqi)

for i in l2:
    i.set_index(pd.date_range("01/2016","12/2022",freq="MS",inclusive="both"),inplace=True)

# %% [markdown]
# # Filling up Nan values

# %%
values = [l2[0].iloc[i,0] for i in range(5,66,6)]
l2[0].iloc[65,0] = np.nanmean(values)
values = [l2[0].iloc[i,0] for i in range(4,65,12)]
l2[-1].iloc[64,0] = np.nanmean(values)

# %% [markdown]
# # Getting AAQ data 

# %%

aaq_data_6_no = pd.read_csv("Datasets/AAQ_Data_2016_1/NOx-Table 1.csv")
aaq_data_6_pm = pd.read_csv("Datasets/AAQ_Data_2016_1/PM10-Table 1.csv")
aaq_data_6_so = pd.read_csv("Datasets/AAQ_Data_2016_1/SO2-Table 1.csv")



aaq_data_7_no = pd.read_csv("Datasets/AAQ_Data_2017_1/NOx-Table 1.csv")
aaq_data_7_pm = pd.read_csv("Datasets/AAQ_Data_2017_1/PM10-Table 1.csv")
aaq_data_7_so = pd.read_csv("Datasets/AAQ_Data_2017_1/SO2-Table 1.csv")



aaq_data_8_no = pd.read_csv("Datasets/AAQ_Data_Jan_-_Dec_2018/NOx-Table 1.csv")
aaq_data_8_pm = pd.read_csv("Datasets/AAQ_Data_Jan_-_Dec_2018/PM10-Table 1.csv")
aaq_data_8_so = pd.read_csv("Datasets/AAQ_Data_Jan_-_Dec_2018/SO2-Table 1.csv")



aaq_data_9_no = pd.read_csv("Datasets/AAQData2019_1/NOx-Table 1.csv")
aaq_data_9_pm = pd.read_csv("Datasets/AAQData2019_1/PM10-Table 1.csv")
aaq_data_9_so = pd.read_csv("Datasets/AAQData2019_1/SO2-Table 1.csv")



aaq_data_20_no = pd.read_csv("Datasets/MonthlyAAQData2020/NOx-Table 1.csv")
aaq_data_20_pm = pd.read_csv("Datasets/MonthlyAAQData2020/PM10-Table 1.csv")
aaq_data_20_so = pd.read_csv("Datasets/MonthlyAAQData2020/SO2-Table 1.csv")



aaq_data_21_no = pd.read_csv("Datasets/MonthlyAAQData2021/NOx-Table 1.csv")
aaq_data_21_pm = pd.read_csv("Datasets/MonthlyAAQData2021/PM10-Table 1.csv")
aaq_data_21_so = pd.read_csv("Datasets/MonthlyAAQData2021/SO2-Table 1.csv")



aaq_data_22_no = pd.read_csv("Datasets/MonthlyAAQData2022/NOx-Table 1.csv")
aaq_data_22_pm = pd.read_csv("Datasets/MonthlyAAQData2022/PM10-Table 1.csv")
aaq_data_22_so = pd.read_csv("Datasets/MonthlyAAQData2022/SO2-Table 1.csv")


# %% [markdown]
# # Organising AAQ data

# %% [markdown]
# ## All L3 are of the form warangal, karimnagar, khammam, nizamabad, adilabad

# %%
l_no = [aaq_data_6_no,aaq_data_7_no,aaq_data_8_no,aaq_data_9_no,aaq_data_20_no,aaq_data_21_no,aaq_data_22_no]
l_pm = [aaq_data_6_pm,aaq_data_7_pm,aaq_data_8_pm,aaq_data_9_pm,aaq_data_20_pm,aaq_data_21_pm,aaq_data_22_pm]
l_so = [aaq_data_6_so,aaq_data_7_so,aaq_data_8_so,aaq_data_9_so,aaq_data_20_so,aaq_data_21_so,aaq_data_22_so]

# %%
for p in [l_no,l_pm,l_so]:
    for i in p:
        i.drop("S.NO",axis = 1,inplace=True)

# %%
for j in [l_no,l_pm,l_so]:
    for i in j:
        for k in i.columns[1:]:
            i[k] = i[k].fillna(i[k].mean(axis=0))

# %%
def arrange(l):
    l3=[]
    for i in range(0,5):
        aaq = pd.DataFrame()
        for j in l:
            aaq = pd.concat([aaq,j.iloc[i,1:]],ignore_index=True)
        l3.append(aaq)

    for i in l3:
        i.set_index(pd.date_range("01/2016","12/2022",freq="MS",inclusive="both"),inplace=True)
    return l3
l3_no = arrange(l_no)
l3_pm = arrange(l_pm)
l3_so = arrange(l_so)

# %%
plt.plot(l3_pm[3])

# %% [markdown]
# # Differencing

# %%
temp_aqi = l2[0] - l2[0].shift(2)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp_aaq = seasonal_decompose(l2[0],model = "additive")
decomp_aaq.seasonal.plot()

# %%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import acf,pacf
acf_aqi = acf(l2[0][0:72])
pacf_aqi = pacf(l2[0][0:72])
plot_acf(acf_aqi)

# %%
exogeneous_ni = np.concatenate((l3_no[3].values,l3_so[3].values,l3_pm[3].values),axis=1)
exogeneous_ad = np.concatenate((l3_no[4].values,l3_so[4].values,l3_pm[4].values),axis=1)
exogeneous_wa = np.concatenate((l3_no[0].values,l3_so[0].values,l3_pm[0].values),axis=1)
exogeneous_ka = np.concatenate((l3_no[1].values,l3_so[1].values,l3_pm[1].values),axis=1)
exogeneous_kh = np.concatenate((l3_no[2].values,l3_so[2].values,l3_pm[2].values),axis=1)

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %% [markdown]
# ## Nizamabad 

# %%
exogeneous_ni[72:]

# %%
rolling_origin = l2[0].iloc[0:72,:].copy()
for i in range(12):
    train_data = l2[0].iloc[0:72+i,:].astype("float64")
    aqi_sarima = SARIMAX(train_data,order = (1,0,0),seasonal_order = (4,0,2,12),exog = exogeneous_ni[0:72+i].astype("float64"))
    aqi_sarima_fit = aqi_sarima.fit()
    
    rolling_origin = pd.concat([rolling_origin,aqi_sarima_fit.forecast(1,exog=exogeneous_ni[72+i:72+i+1].astype("float64"))],ignore_index=False)
    

# %%
aqi_sarima = SARIMAX(rolling_origin.astype("float64"),order = (1,0,0),seasonal_order = (4,0,2,12))
aqi_sarima_fit = aqi_sarima.fit()

plt.plot(aqi_sarima_fit.forecast(12))

plt.plot(l2[0],c="orange")
#plt.plot(rolling_origin,c="blue")
plt.plot(aqi_sarima_fit.forecast(12))

# %%
print("The prediction:\n")
print(aqi_sarima_fit.forecast(12))

# %% [markdown]
# ## warangal

# %%
rolling_origin = l2[2].iloc[0:72,:].copy()
for i in range(12):
    train_data = l2[2].iloc[0:72+i,:].astype("float64")
    aqi_sarima = SARIMAX(train_data,order = (1,0,0),seasonal_order = (4,0,2,12),exog = exogeneous_wa[0:72+i].astype("float64"))
    aqi_sarima_fit = aqi_sarima.fit()
    
    rolling_origin = pd.concat([rolling_origin,aqi_sarima_fit.forecast(1,exog=exogeneous_wa[72+i:72+i+1].astype("float64"))],ignore_index=False)

# %%
plt.plot(l2[2],c="orange")
plt.plot(rolling_origin.tail(12),c="blue")

# %%
aqi_sarima = SARIMAX(rolling_origin.astype("float64"),order = (1,0,0),seasonal_order = (4,0,2,12))
aqi_sarima_fit = aqi_sarima.fit()

plt.plot(aqi_sarima_fit.forecast(12))

plt.plot(l2[2],c="orange")
#plt.plot(rolling_origin,c="blue")
plt.plot(aqi_sarima_fit.forecast(12),c="blue")

# %%
print("The predictions:\n")
print(aqi_sarima_fit.forecast(12))

# %% [markdown]
# ## karimnagar

# %%
rolling_origin = l2[3].iloc[0:72,:].copy()
for i in range(12):
    train_data = l2[3].iloc[0:72+i,:].astype("float64")
    aqi_sarima = SARIMAX(train_data,order = (1,1,0),seasonal_order = (4,0,2,12),exog = exogeneous_ka[0:72+i].astype("float64"))
    aqi_sarima_fit = aqi_sarima.fit()
    
    rolling_origin = pd.concat([rolling_origin,aqi_sarima_fit.forecast(1,exog=exogeneous_ka[72+i:72+i+1].astype("float64"))],ignore_index=False)

# %%
plt.plot(l2[3],c="orange")
plt.plot(rolling_origin.tail(12),c="blue")

# %%
aqi_sarima = SARIMAX(rolling_origin.astype("float64"),order = (1,0,0),seasonal_order = (4,0,2,12))
aqi_sarima_fit = aqi_sarima.fit()

plt.plot(aqi_sarima_fit.forecast(12))

plt.plot(l2[3],c="orange")
#plt.plot(rolling_origin,c="blue")
plt.plot(aqi_sarima_fit.forecast(12),c="blue")

# %%
print("The predictions:\n")
print(aqi_sarima_fit.forecast(12))

# %% [markdown]
# ## Khammam

# %%
rolling_origin = l2[4].iloc[0:72,:].copy()
for i in range(12):
    train_data = l2[4].iloc[0:72+i,:].astype("float64")
    aqi_sarima = SARIMAX(train_data,order = (1,2,0),seasonal_order = (4,0,2,12),exog = exogeneous_kh[0:72+i].astype("float64"))
    aqi_sarima_fit = aqi_sarima.fit()
    
    rolling_origin = pd.concat([rolling_origin,aqi_sarima_fit.forecast(1,exog=exogeneous_kh[72+i:72+i+1].astype("float64"))],ignore_index=False)

# %%
plt.plot(l2[4],c="orange")
plt.plot(rolling_origin.tail(12),c="blue")

# %%
np.sqrt(((l2[4].iloc[72:,0] -  rolling_origin.iloc[72:,0])**2).sum())

# %%
aqi_sarima = SARIMAX(rolling_origin.astype("float64"),order = (1,1,0),seasonal_order = (4,0,2,12))
aqi_sarima_fit = aqi_sarima.fit()

plt.plot(aqi_sarima_fit.forecast(12))

plt.plot(l2[3],c="orange")
#plt.plot(rolling_origin,c="blue")
plt.plot(aqi_sarima_fit.forecast(12),c="blue")

# %%
print("The prediction:\n")
print(aqi_sarima_fit.forecast(12))

# %% [markdown]
# ## Adilabad

# %%
from impyute.imputation.cs import mice

# %%
ind = np.concatenate(((l2[1].values.astype('float64')), (exogeneous_ad.astype('float64'))), axis = 1)

# %%
imp = mice(ind)

# %%
plt.plot(pd.date_range("01/2016","12/2022",freq="MS",inclusive="both"), imp[:,0])

# %%
from statsmodels.tsa.arima.model import ARIMA


# %%
l2[1][0] = imp[:,0]

# %%
l2[1][0].isnull().sum()

# %%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import acf,pacf
acf_aqi = acf(l2[1][0:72])
pacf_aqi = pacf(l2[1][0:72])
plot_acf(acf_aqi)

# %%
rolling_origin = l2[1].iloc[0:72,:].copy()
for i in range(12):
    train_data = l2[1].iloc[0:72+i,:].astype("float64")
    aqi_sarima = SARIMAX(train_data,order = (1,1,1),seasonal_order = (4,2,4,12), exog = exogeneous_ad[0:72+i].astype("float64"))
    aqi_sarima_fit = aqi_sarima.fit()
    
    rolling_origin = pd.concat([rolling_origin,aqi_sarima_fit.forecast(1,exog=exogeneous_ad[72+i:72+i+1].astype("float64"))],ignore_index=False)

# %%
train_data = l2[1].iloc[0:72,:].astype("float64")
aqi_sarima = SARIMAX(rolling_origin,order = (1,1,1),seasonal_order = (4,2,4,12))
aqi_sarima_fit = aqi_sarima.fit()

# %%
forecast = aqi_sarima_fit.forecast(12)#exog=exogeneous_ad[72:].astype("float64"))

# %%
print("The prediction:\n")
print(forecast)

# %%
plt.plot(l2[1],c="orange")
plt.plot(forecast,c="blue")


