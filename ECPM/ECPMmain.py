import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from math import exp
from numpy import log
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from matplotlib.pylab import rcParams

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = 'plotly_white'

# Preferred settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)


data = pd.read_csv('data/CarbonEmissionIndia.csv')
data2 = pd.read_csv('data/india-carbon-co2-emissions.csv')

df = data
df2 = data2

df.head()

# Only include data from 1950 and later
#df = df[df['year']>=1950]

#!df.dropna(axis=1,thresh=8000, inplace=True)

# Drop continents, identified by lacking an ISO code
#df.dropna(axis=0,subset=['iso_code'],inplace=True)

# Drop "World" entries
#df = df[df.country != 'World']

# Observe new df shape
#print(df.shape)










# Visualize top states by CO,C02 and CH4 emissions by states
co2States = df.groupby(['States'])['co2'].mean().sort_values(ascending=False).index[:29]
coStates = df.groupby(['States'])['co'].mean().sort_values(ascending=False).index[:29]
ch4States = df.groupby(['States'])['ch4'].mean().sort_values(ascending=False).index[:29]
height = df.groupby(['States'])['co2'].mean().sort_values(ascending=False)[:29]

x=co2States
y=coStates
z=ch4States


fig, ax = plt.subplots(figsize=(15,10))
ax.set_facecolor("black")
plt.bar(x, height=height, color = 'mediumseagreen')
plt.bar(y, height=height, color = 'red')
plt.bar(z, height=height, color = 'blue')
ax.legend(labels=['co2', 'co','ch4'])

plt.title("Mean CO2,CO AND CH4 Emissions per states", fontsize=20)
plt.ylabel('Mean Total Emissions Per states (Tonnes)', fontsize=15)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
    tick.label.set_rotation('vertical')

# Threshold to indicate countries above 150 mean tonnes
threshold = 500
plt.axhline(y=threshold,linewidth=2, linestyle='dashed', color='white')

plt.show()

# Uncomment below code only to save image
#plt.savefig("images/mean_emissions_per_year_top50")





# Find top 10 emitters per capita

# Create list of top 10 emitters
# top_emit = list(df.groupby(['States'])['total'].mean().sort_values(ascending=False)[:22].index)
# df_top_emit = df.loc[df['States'].isin(top_emit)]
# top10_percapita = list(df_top_emit.groupby(['States'])['total'].mean().sort_values(ascending=False)[:10].index)

# # Plot data
# df_top10 = df.loc[df['States'].isin(top10_percapita)]
# fig, ax = plt.subplots(figsize=(15,6))
# plt.bar(x = top10_percapita,
#         height = df_top10.groupby(['States'])['total'].mean().sort_values(ascending=False),
#        color = 'royalblue')

# plt.title('Top 10  Emitters Per Capita \n 1990-2020', fontsize=15)
# plt.ylabel('Mean Emissions Per Year Per Capita (Tonnes)', fontsize=12)
# plt.xlabel('States', fontsize=12)


#using second csv file which holds data of CO2 emissions in India since 1990
#pre-prediction calculations starts






# Only include india co2_per_capita data 
# df2 = df2[df2['year'] >= 1990]
df2 = df2[['year2','co2capita']]
# Convert year to DateTime object
df2['year2'] = pd.to_datetime(df2['year2'],format='%Y')

# Convert year to index
df2.set_index('year2', inplace=True)

# Observe updated df
df2.head()


# Visualize India's CO2 emissions per capita since 1990,
# including 5-year rolling mean and rolling standard deviation
# df2.index = df2.index.map(str)

roll_mean = df2.rolling(window=5, center=False).mean()
roll_std = df2.rolling(window=5, center=False).std()
fig, ax = plt.subplots(figsize=(5,5))
ax.set_facecolor("black")
plt.plot(df2,color='yellow', label='Original')
plt.plot(roll_mean, color='maroon', label='Rolling Mean')
plt.plot(roll_std, color='white', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.xlabel('Year')
plt.ylabel('CO2 Per Capita (Tonnes)')
plt.show(block=False)

# Dickey-Fuller test
test = adfuller(df2['co2capita'][1:-1])
dfoutput = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput)
#When we make a model for forecasting purposes in time series analysis, we require a stationary time series for better prediction. So the first step to work on modeling is to make a time series stationary. Testing for stationarity is a frequently used activity in autoregressive modeling. We can perform various tests like the KPSS, Phillips–Perron, and Augmented Dickey-Fuller.
#Result : Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.


#to make a prediction model we need to render the data in such a way
#that it dosent exhibit any trend, hence is stationary



# Log transformation to address lack of variance and covariance
# Create new df that contains the logged values of the original one
log_IND = log(df2)

# Visualize logged data, including 5-year rolling mean and standard deviation
roll_mean_log = log_IND.rolling(window=5, center=False).mean()
roll_std_log = log_IND.rolling(window=5, center=False).std()
fig, ax = plt.subplots(figsize=(5,5))
ax.set_facecolor("black")
plt.plot(log_IND,color='mediumseagreen', label='Original')
plt.plot(roll_mean_log, color='mediumblue', label='Rolling Mean')
plt.plot(roll_std_log, color='orangered', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation, Logged Data')
plt.xlabel('Year')
plt.ylabel('Logged CO2 Per Capita (Tonnes)')
plt.show(block=False)

# Dickey-Fuller test
test = adfuller(log_IND['co2capita'][1:-1])
dfoutput = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput)

    #as high p-value can be observed again , we need to try different method
    #to achieve stationarity
    
    
    
    
log_minus_rollmean = log_IND - roll_mean_log

# Drop null values
log_minus_rollmean.dropna(axis=0,inplace=True)
fig, ax = plt.subplots(figsize=(5,5))
#ax.set_facecolor("black")
# Plot data
plt.plot(log_minus_rollmean, color='red')
plt.title('Logged Data Minus Rolling Mean')
plt.xlabel('Year')
plt.ylabel('Logged CO2 Per Capita Minus Rolling Mean')
plt.show(block=False)

# Dickey-Fuller test on logged data minus rolling mean
test = adfuller(log_minus_rollmean['co2capita'][1:-1])
dfoutput = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput)    



#This data looks much more stationary based on its plot and its Dickey-Fuller
# test also yielded a very low p-value, indicating that we can confidently
# reject the null hypothesis that the data is non-stationary.


#Now that we have produced a logged will now difference our
# original data until we achieve stationarity so we can 
#model on both datasets.

# Difference original df
# diff = df2.diff().rename(index=str, columns={"co2capita": "Differenced Observations"})

# Visualize differenced data, including 5-year rolling mean and standard deviation
# roll_mean_diff = diff.rolling(window=5, center=False).mean()
# roll_std_diff = diff.rolling(window=5, center=False).std()
# fig, ax = plt.subplots(figsize=(5,5))
# plt.plot(diff,color='mediumseagreen', label='Original')
# plt.plot(roll_mean_diff, color='mediumblue', label='Rolling Mean')
# plt.plot(roll_std_diff, color='orangered', label = 'Rolling Std')
# plt.legend(loc='best')
# plt.title('Rolling Mean & Standard Deviation, Differenced Data')
# plt.xlabel('Differenced Year')
# plt.ylabel('Differenced CO2 Per Capita (Tonnes)')
# plt.show(block=False)

# # Dickey-Fuller test on differenced data
# test = adfuller(diff['Differenced Observations'][1:-1])
# dfoutput = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
# print(dfoutput)


#We will now create time series models on both our logged data and original data to see which yields better results.
# ACF for logged data


plot_acf(log_IND[:-1], alpha=.05);

# PACF for logged data
plot_pacf(log_IND[:-1], alpha=.05, lags=20);

# Instantiate ARIMA model for logged data

# Instantiate model 1
mod_log = ARIMA(log_IND[:-1], order=(2,2,1))

# Fit Model
mod_log_fit = mod_log.fit()

# Obeserve summary statistics
print(mod_log_fit.summary())


#to determine best value of parameters we find out which
#parameters gives the least value for ACF and BCF
#library pdarima offers auto determination of order, optimizeOrder.py has been used to determine the best possible order of the dataset

# ACF for original data
plot_acf(df2[:-1], alpha=.05);

# PACF for original data
plot_pacf(df2[:-1], alpha=.05, lags=20);


# Instantiate model
mod221_fit = ARIMA(df2[:-1], order=(2,2,1))

# Fit model
mod221_fit = mod221_fit.fit()

# Observe summary statistics
print(mod221_fit.summary())

# Train-test split

X = log_IND.index
y = log_IND['co2capita']

train = log_IND.loc[:'2009-01-01']
test = log_IND.loc['2009-01-01':]

X_train, X_test = train.index , test.index
y_train, y_test = train['co2capita'] , test['co2capita']
# Best model:  ARIMA(2,2,1)(0,0,0)[0] 
mod_logAR = AutoReg(train[2:], lags=2)
mod_logAR_fit = mod_logAR.fit()

# Prediction
pred_log = mod_log_fit.predict(start='2009-01-01', end='2018-01-01', dynamic=False)

#AR model
# Instantiate and fit model to train data
mod_logAR = AutoReg(train[2:], lags=2)
mod_logAR_fit = mod_logAR.fit()

# Prediction
pred_logAR = mod_logAR_fit.predict(start='2009-01-01', end='2018-01-01', dynamic=False)


#now on orignal data
# Train-test split

X = df2.index
y = df2['co2capita']

train = df2.loc[:'2009-01-01']
test = df2.loc['2009-01-01':]

X_train, X_test = train.index , test.index
y_train, y_test = train['co2capita'] , test['co2capita']

mod221 = ARIMA(y_train[2:], order=(2,2,1)) # p,d,q
mod221_fit = mod221.fit()

#Prediction
pred221 = mod221_fit.predict(start='1997-01-01', end='2018-01-01', dynamic=False)

mod_logAR_fit.plot_predict();


pred_correctunits = np.e**(pred_logAR)

# Plot real vs predicted values
rcParams['figure.figsize'] = 15, 6

# Plot observed values
ax = df2['1950-01-01':].plot(label='observed', color='royalblue')

# Plot predicted values
pred_correctunits.plot(ax=ax, label='Forecast', alpha=0.9, color='orangered')

# Set axes labels and title
ax.set_xlabel('Year', size=12)
ax.set_ylabel('CO2 Emissions (Tonnes)',size=12)
ax.set_title('Real vs Predicted CO2 Emissions Per Capita', size=15)

plt.legend()
plt.show()





