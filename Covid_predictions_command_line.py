#!/usr/bin/env python
# coding: utf-8

# In[12]:


import warnings
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

from pmdarima import AutoARIMA
import pmdarima as pm

import plotly.express as px

#from datetime import timedelta
from datetime import date, timedelta, datetime


sns.set()
warnings.simplefilter('ignore')


# In[18]:


df = pd.read_csv('./data/countries-aggregated.csv')


df = df.dropna()
df


# In[5]:


df.Country.unique()


# In[7]:


# Making time-series data
def time_series(country):
    '''Return a time-series dataframe with new_deaths and new_cases'''
    data = df[df['Country'] == f'{country}']
    data['new_deaths'] = data['Deaths'] - data['Deaths'].shift(1)
    data['new_cases'] = data['confirmed'] - data['confirmed'].shift(1)
    # Making time-series data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data.dropna()


# In[19]:


# Time-series plot
def plot_timeseries(case, country):
    '''Plot a time series for 'Confirmed', 'Recovered', 'Deaths', 'new_deaths',
       'new_cases' for any country '''
    f, ax = plt.subplots(figsize=(16, 5))
    labels = {country}
    country1 = time_series(country)

    plt.plot(country1.index, country1[case].values, linewidth=2)
    plt.xticks(country1.resample('M').min().index)
    plt.legend(labels)
    ax.set(
        title=f' Evolution of actual {case} in {country}',
        ylabel='Number of cases')

    plt.legend(labels)
    plt.show()

    #df_by_date = pd.DataFrame(data.fillna('NA').groupby(['Country','Date'])['Confirmed'].sum().sort_values().reset_index())

    f = px.bar(
        country1.sort_values(
            case,
            ascending=False),
        x=country1.index,
        y=country1[case].values,
        color=f'{case}',
        labels=['Case'])
    f.update_layout(title_text=f'COVID-19 {case} per day in {country}')
    f.show()


# In[120]:


now = datetime.now()  # current date and time
date_today = now.strftime("%Y-%m-%d")

# fit stepwise auto-ARIMA


def Arima_fit(case, country, start_date='2020-11-10', end_date='2020-11-17'):
    '''Use AutoArima to find the best Arima model to fit the train data and return
    prediction for dates between start_date and end_date. Please use start date as today'''
    country2 = time_series(country)

    forecast_date = pd.date_range(start_date, end_date)
    values_fit = np.log(
        country2[case].loc['2020-03-01':date_today].dropna().values)
    values_fit_positive = values_fit[values_fit > 0]

    stepwise_fit = pm.auto_arima(values_fit_positive, start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=7,
                                 start_P=0, seasonal=True,
                                 d=1, D=1, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)  # set to stepwise

    stepwise_fit.fit(values_fit_positive)
    y_pred = stepwise_fit.predict(len(forecast_date))
    y_pred_df = pd.DataFrame({f'{case}': np.exp(y_pred)}, index=forecast_date)

    y_case = pd.DataFrame({f'{case}': country2[case].loc['2020-03-01':].dropna(
    ).values}, index=country2[case]['2020-03-01':].index)
    y_pred_df.to_csv("data/predictions_Germany.csv", index=False)

    return (y_pred_df, y_case)


# In[103]:


def plot_prediction(case, country, start_date, end_date):
    '''plot predictions using AutoArima to find the best Arima model to fit the train data and return
    prediction for dates between start_date and end_date. Please use start date as today'''

    list_df = Arima_fit(case, country, start_date, end_date)
    fig, ax = plt.subplots()
    list_df[1]['new_cases'].loc['2020-03-01':].plot(ax=ax)
    list_df[0].plot(ax=ax)

    fig.set_size_inches(15, 10)
    fig.savefig(f"output_{country}.png")

    return list_df


if __name__ == "__main__":

    case_input = input(
        'choose between Confirmed, Recovered, Deaths, new_deaths, new_cases:')
    country_input = input(
        'Give me a name of a Country, you would like to check Covid-19 analysis: ')

    plot_timeseries(case_input, country_input)
    case = input(
        'choose between confirmed, recovered, Deaths, new_deaths, new_cases: ')
    country = input(
        'Give me a name of a Country, you would like to check Covid-19 analysis: ')
    starting_date = input(
        'Give me a starting date in the format 2020-month-date: ')
    ending_date = input('Give me a end date in the format 2020-month-date: ')
    plot_prediction(case, country, starting_date, ending_date)
