from pmdarima import auto_arima
import pandas as pd
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
import warnings
warnings.filterwarnings("ignore")


optData = pd.read_csv('data/india-carbon-co2-emissions.csv')
df = optData

stepwise_fit = auto_arima(df['co2capita'], trace= True,suppress_warnings=True)
stepwise_fit.summary()



