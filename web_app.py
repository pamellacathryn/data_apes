import streamlit as st
import altair as alt
import plotly.express as px
import yfinance as yf
from pandas_datareader import data
import datetime
from streamlit_option_menu import option_menu
from warnings import simplefilter
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.metrics import mean_squared_error
from scipy.signal import periodogram
from pandas_datareader.data import DataReader
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

stopper = False

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


def create_models(fourier_pairs, target, kode_emiten):
    fourier = CalendarFourier(freq="A", order=fourier_pairs)

    dp = DeterministicProcess(
        index=datumz.index,
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

    perc = (target_fore[0] * 100 / target[-1]) - 100

    return (target, target_pred, target_fore, perc)


def create_plot(target, target_pred, target_fore, ax, kode):
    target.plot(ax=ax)
    target_pred.plot(ax=ax)
    target_fore.plot(ax=ax)
    ax.set_title(f'Saham {kode.upper()}')
    ax.set_ylabel('Close Price')
    ax.legend(['Price', 'Model Forecasts'])
    plt.show()
    return


def check_pred(target, target_fore, perc):
    if target[-1] > 5000:
        cap = 20
    elif target[-1] > 200:
        cap = 25
    else:
        cap = 35

    if np.abs(perc) > cap and perc > 0:
        perc = cap
        target_fore = target[-1] * (100 + cap)
    elif np.abs(perc) > cap and perc > 0:
        perc = cap * -1
        target_fore = target[-1] * (100 + cap)

    return (target_fore, perc)

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

  return (datas['Close'].values[-1], target_fore[0], percentage_data['Percentage (%)'].values[0])

df = pd.read_table("Saham per Sektor(3).csv",delimiter=",")

st.set_page_config(
    page_title="Saham Web App",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "This Web App is made by Data Apes Team for TSDN 2022"
    }
)

with st.sidebar:
    st.header("Navigation")
    select = option_menu(
        menu_title=None,
        options=["Main","Stock List"],
        icons=["cash-stack","filter-left"],
        styles={"nav-link":{"font-size":"13px"}}
    )
    st.header("About")
    st.info("This web app is made by [Ariabagus](https://www.linkedin.com/in/bagood/), [Dira](https://www.linkedin.com/in/muhammad-dira-kurnia-a7a749186/), [Kevin](https://www.linkedin.com/in/kevin-sean/), [Pamella](https://www.linkedin.com/in/pamellacathryn/), and [Yohanes](https://www.linkedin.com/in/yohanesyordan/).")

if select == "Main":
    st.markdown("<h1 style='text-align: center; '>Your Stock Analysis Partner</h1>",
                unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>Data Apes</h6>",
                unsafe_allow_html=True)
    st.write("")
    st.header('Start Your Analysis and Create Predictions!')
    koloms1, koloms2, koloms3 = st.columns([1,1,1])
    profil = koloms1.radio("Your Investment Type",('Short Term', 'Long Term'))
    output = koloms1.radio("Output",('Stock Price Prediction','Top Gainer Stocks'))
    koloms3.write("")

    if output == "Top Gainer Stocks":
        sector = st.selectbox(
            'Choose the Sector:',
            ('Sectors 👇🏼', 'Basic Materials', 'Consumer Cyclicals', 'Consumer Non-Cyclicals', 'Energy',
     'Financials', 'Healthcare', 'Industrials', 'Infrastructures',
     'Properties & Real Estate', 'Technology', 'Transportation & Logistic'))
    elif output == "Stock Price Prediction":
        emiten = st.text_input('Input Stock Index (Ex: BBCA, bbca)',"")
        emiten = emiten.upper()
        st.write("")

    if output == "Stock Price Prediction":
        emiten_jk = emiten+'.JK'
        if emiten_jk != ".JK":
            symbol = yf.Ticker(emiten_jk).info

            if symbol.get('longName') is not None:
                kol1, kol2 = st.columns([1,5])
                kol1.image(symbol.get('logo_url'))
                kol2.markdown(f"<h3 style='text-align: left; '>{symbol.get('longName')}</h3>", unsafe_allow_html=True)
                kol2.write(f"Country: {symbol.get('country')}")
                kol2.write(f"Currency: {symbol.get('currency')}")
                index = df.index[df['Kode'] == emiten_jk].tolist()
                sector = df.iloc[index]["Tipe"].to_list()
                sector = sector[0]
                df_sector = df[(df == sector).any(axis=1)]
                sector_tickers = df_sector["Kode"].to_list()
                kol2.write(f"Sector: {sector}")
                kol2.write(f"Website: {symbol.get('website')}")
                kol2.markdown(f"<div style='text-align: justify;'>{symbol.get('longBusinessSummary')}</div>", unsafe_allow_html=True)
                st.write("")

                if profil == "Short Term":
                    st.markdown(f"<h4 style='text-align: center; '>Stock Price Forecasting of {emiten}</h4>",
                                unsafe_allow_html=True)
                    col1, col2 = st.columns([3,1])
                    start = col2.date_input(
                        "Start Date:",
                        datetime(2022, 11, 1))

                    datumz = yf.download(emiten_jk, start)
                    _, ax = plt.subplots()
                    try:
                        datumz = datumz.to_period('D')
                        target, target_pred, target_fore, perc = create_models(12, datumz['Close'], emiten_jk)
                        target_fore, perc = check_pred(target, target_fore, perc)
                        aw = create_plot(target, target_pred, target_fore, ax, emiten_jk)
                    except:
                        pass

                    histor = datumz[['Close']]
                    histor.columns = [emiten_jk]
                    forecastz = target_fore[0]
                    forecastz = pd.DataFrame([forecastz])
                    after_forecastz = pd.concat([datumz["Close"], forecastz])
                    b = after_forecastz.index[-2]
                    b += timedelta(days=1)
                    after_forecastz["index1"] = after_forecastz.index
                    after_forecastz["index1"].iloc[-1] = b
                    after_forecastz.index = list(after_forecastz["index1"])
                    after_forecastz = after_forecastz.drop(columns=["index1"])
                    gabung = pd.concat([after_forecastz, histor], axis=1)
                    gabung.columns = [emiten+"_Forecast",emiten]
                    col1.line_chart(gabung)
                    kenaikan = (((target_fore[0]-histor.iloc[-1].to_numpy())/histor.iloc[-1].to_numpy())*100)
                    for k in kenaikan:
                        kenaikan = k
                    col2.write(f"Increment (%): {round(kenaikan,2)}%")

            else:
                st.error('Stock Not Found', icon="🚨")

    elif output == "Top Gainer Stocks":
        if sector != "Sectors 👇🏼":
            if profil == "Short Term":

                sektor = df.loc[df['Tipe'] == sector, :]
                kode = sektor['Kode'].values
                start = st.date_input(
                                "Start Date:",
                                datetime(2022, 11, 1))
                end = datetime.now()
                st.write("")
                with st.spinner('Wait for it... (it might take a while)'):
                    result = pd.DataFrame(
                        {'Kode Perusahaan': [], 'Peningkatan (%)': [], 'Harga Terakhir': [], 'Harga Prediksi': []})

                    for k in kode:
                        datumz = yf.download(k, start, end)

                        try:
                            datumz = yf.download(k, start)
                            datumz = datumz.to_period('D')
                            target, target_pred, target_fore, perc = create_models(12, datumz['Close'], k)
                            target_fore, perc = check_pred(target, target_fore, perc)
                            app = pd.DataFrame({'Kode Perusahaan': [k], 'Peningkatan (%)': [perc],
                                                'Harga Terakhir': [target[-1]], 'Harga Prediksi': [target_fore[0]]})
                            result = pd.concat([result, app])
                        except:
                            pass
                    result = result.sort_values('Peningkatan (%)', ascending=False)
                    result = result[0:10]
                    result = result.set_index([pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])
                    st.markdown(f"<h4 style='text-align: center; '>Top 10 Stock's Growth Percentage on {sector}</h4>",
                                unsafe_allow_html=True)
                    for i in range(10):
                        result = result.replace([result.iloc[i]["Kode Perusahaan"]], result.iloc[i]["Kode Perusahaan"].replace('.JK', ''))
                    st.table(result[["Kode Perusahaan","Peningkatan (%)"]])
                    result = result.reset_index()

                st.markdown(f"<h4 style='text-align: center; '>Stock Prices Forecasting on {sector}</h4>",
                            unsafe_allow_html=True)
                top_10 = yf.download(result["Kode Perusahaan"].iloc[0], start)["Close"]
                forecazt = []
                for i in range(10):
                    forecazt.append(result[["Harga Prediksi"]].to_numpy()[i][0])
                for k in result["Kode Perusahaan"]:
                    k = k+'.JK'
                    datay = yf.download(k, start)
                    tampung = datay["Close"]
                    top_10 = pd.concat([top_10, tampung], axis=1)
                top_10 = top_10.iloc[:, 1:]
                top_10.columns = result["Kode Perusahaan"]
                terakhir = top_10.iloc[-1]
                nama_koloms = top_10.columns
                nama_koloms_baru = []
                for i in nama_koloms:
                    nama_koloms_baru.append(i + '_Forecast')
                for i in range(10):
                    terakhir[i] = forecazt[i]
                forecast = pd.DataFrame(terakhir).transpose()
                a = forecast.index
                a = a.date + pd.Timedelta(days=1)
                forecast = forecast.set_index(a)
                after_forecast = pd.concat([top_10, forecast])
                after_forecast.columns = nama_koloms_baru
                gabung = pd.concat([after_forecast,top_10], axis=1)

                colz1, colz2 = st.columns([3, 1])
                tampil = colz2.multiselect(
                    'Tampilkan:',
                    result["Kode Perusahaan"])
                tampilkan = []
                for i in tampil:
                    tampilkan.append(i+"_Forecast")
                    tampilkan.append(i)
                tampil_chart = gabung[tampilkan]
                colz1.line_chart(tampil_chart)

elif select == "Stock List":
    st.header("Indonesia Stock Indexes")
    df_nonjk = df.copy()
    for i in range(len(df_nonjk)):
        string_1 = df_nonjk.iloc[i]["Kode"]
        string_2 = ".JK"
        if string_2 in string_1:
            baru = string_1.replace(string_2, '')
        df_nonjk.iloc[i]["Kode"] = baru
    st.table(df_nonjk)


