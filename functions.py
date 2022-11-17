import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import yfinance as yf
import datetime
from streamlit_option_menu import option_menu
from warnings import simplefilter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.metrics import mean_squared_error
from scipy.signal import periodogram
from pandas_datareader.data import DataReader
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np


def create_and_plot_models(fourier_pairs, target, ax, title, y_label):
    fourier = CalendarFourier(freq="A", order=fourier_pairs)

    dp = DeterministicProcess(
        index=datas.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )

    X = dp.in_sample()
    linreg_model = LinearRegression(fit_intercept=False)
    linreg_model.fit(X, target)
    target_pred = pd.Series(linreg_model.predict(X), index=target.index)
    features_fore = dp.out_of_sample(steps=1)
    target_fore = pd.Series(linreg_model.predict(features_fore), index=features_fore.index)
    rmse = mean_squared_error(target, target_pred, squared=False)
    if ax != None:
        target.plot(ax=ax)
        target_pred.plot(ax=ax)
        target_fore.plot(ax=ax)
        ax.set_title(f'{title}\nPredictions RMSE: {rmse}')
        ax.set_ylabel(y_label)
        ax.legend(['Price', 'Model Predictions', 'Model Forecasts'])
    return target_fore

def calc_perc_up_down(target, target_fore):
    last_data = target.values[-1]
    last_date = datas.index[-1]
    dates = [last_date + timedelta(days=i) for i in range(1, 2)]
    perc = []
    for i in range(0, len(target_fore)):
        calc = (target_fore[i] * 100 / last_data) - 100
        perc.append(calc)
    percentage = pd.DataFrame({'Date': dates, 'Percentage (%)': perc})
    return percentage

def execute(datas, company_code, ax):
  target_fore = create_and_plot_models(12, datas['Close'], ax, 'TCS High Stock Price - Seasonal Forecast', 'High Price')
  percentage_data = calc_perc_up_down(datas['Close'], target_fore)
  return percentage_data['Percentage (%)'].values[0]

def get_chart(data):
    data["Close"] = round(data["Close"], 2)
    hover = alt.selection_single(
        fields=["Date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    lines = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x="Date",
            y=alt.Y("Close", title="Close"),
        )
    )
    points = lines.transform_filter(hover).mark_circle(size=65)
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="Date",
            y="Close",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date", title="Date"),
                alt.Tooltip("Close", title="Close"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()