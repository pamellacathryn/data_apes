import streamlit as st
import altair as alt
import plotly.express as px
import yfinance as yf
from pandas_datareader import data
import datetime
from streamlit_option_menu import option_menu
from warnings import simplefilter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.metrics import mean_squared_error
from scipy.signal import periodogram
from pandas_datareader.data import DataReader
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

df = pd.read_table("Saham per Sektor.csv",delimiter=",")

st.set_page_config(
    page_title="Saham Web App",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
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
    st.markdown("<h2 style='text-align: center; '>Pemanfaatan Machine Learning dalam Memprediksi Nilai dan Valuasi Saham Indonesia secara Teknikal dan Fundamental</h1>",
                unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>by Data Apes</h6>",
                unsafe_allow_html=True)
    st.write("")
    st.header('Latar Belakang')
    st.markdown('<div style="text-align: justify;">Sebagai investor, tentunya tujuan utama yang diraih adalah keuntungan. Untuk itu, pentingnya untuk mengenali saham yang akan dibeli melalui berbagai analisis saham, salah duanya adalah analisis Teknikal dan Fundamental.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">Analisis teknikal, yang biasanya digunakan untuk melakukan investasi jangka pendek, didasarkan pada data-data harga historis dimana prediksi untuk membeli atau menjual saham dilakukan dengan melihat grafik historis pergerakan saham.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">Analisis fundamental, dimana umum digunakan untuk melakukan investasi jangka panjang, didasarkan oleh kondisi suatu perusahaan, kondisi ekonomi dan industri terkait menggunakan  indikator-indikator perusahaan yang tertera melalui laporan keuangan perusahaan seperti Earnings per share (EPS), Price to Earning Ratio (P/E), dan lain-lainnya.</div>', unsafe_allow_html=True)
    st.markdown("""---""")
    st.header('Pilih-Pilih:')
    koloms1, koloms2, koloms3 = st.columns([1,1,1])
    profil = koloms1.radio("Profil Investor:",('Jangka Pendek', 'Jangka Panjang'))
    output = koloms2.radio("Output:",('Nilai Saham','Rekomendasi'))
    koloms3.write("")

    if output == "Rekomendasi":
        sector = st.selectbox(
            'Pilih Sektor:',
            ('Sectors 👇🏼', 'Basic Materials', 'Consumer Cyclicals', 'Consumer Non-Cyclicals', 'Energy',
     'Financials', 'Healthcare', 'Industrials', 'Infrastructures',
     'Properties & Real Estate', 'Technology', 'Transportation & Logistic'))
    elif output == "Nilai Saham":
        emiten = st.text_input('Kode Perusahaan (Contoh: BBCA)',"")
        emiten = emiten.upper()
        emiten = emiten+'.JK'
        st.write("")

    if output == "Nilai Saham":
        if emiten != ".JK":
            symbol = yf.Ticker(emiten).info

            if symbol.get('longName') is not None:
                kol1, kol2 = st.columns([1,5])
                kol1.image(symbol.get('logo_url'))
                kol2.markdown(f"<h3 style='text-align: left; '>{symbol.get('longName')}</h3>", unsafe_allow_html=True)
                kol2.write(f"Country: {symbol.get('country')}")
                kol2.write(f"Currency: {symbol.get('currency')}")
                index = df.index[df['Kode'] == emiten].tolist()
                sector = df.iloc[index]["Tipe"].to_list()
                sector = sector[0]
                df_sector = df[(df == sector).any(axis=1)]
                sector_tickers = df_sector["Kode"].to_list()
                kol2.write(f"Sector: {sector}")
                kol2.write(f"Website: {symbol.get('website')}")
                kol2.markdown(f"<div style='text-align: justify;'>{symbol.get('longBusinessSummary')}</div>", unsafe_allow_html=True)
                st.write("")

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
                
                if profil == "Jangka Pendek":
                    st.markdown(f"<h4 style='text-align: center; '>Stock Price of {emiten}</h4>",
                                unsafe_allow_html=True)
                    col1, col2 = st.columns([3,1])
                    source = 'yahoo'
                    start = col2.date_input(
                        "Start Date:",
                        datetime(2022, 1, 1))
                    end = col2.date_input(
                        "End Date:")
                    data = data.DataReader(emiten, start=start, end=end, data_source=source).reset_index()
                    # st.write(data)
                    chart = get_chart(data)
                    col1.altair_chart(
                        (chart).interactive(),
                        use_container_width=True)
                    st.write("")

            else:
                st.error('Emiten tidak ditemukan', icon="🚨")

    elif output == "Rekomendasi":
        if sector != "Sectors 👇🏼":
            if profil == "Jangka Pendek":

                sektor = df.loc[df['Tipe'] == sector, :]
                kode = sektor['Kode'].values
                end = datetime.now()
                start = st.date_input(
                                "Peningkatan dari:",
                                datetime(2022, 1, 1))
                with st.spinner('Wait for it...'):

                    result = pd.DataFrame({'Kode Perusahaan': [], 'Rata-rata Peningkatan (%)': []})
                    for k in kode:
                        datas = yf.download(k, start, end)
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
                            target_fore = create_and_plot_models(12, datas['Close'], ax,
                                                                 'TCS High Stock Price - Seasonal Forecast', 'High Price')
                            percentage_data = calc_perc_up_down(datas['Close'], target_fore)
                            return percentage_data['Percentage (%)'].values[0]
                        try:
                            datas = datas.to_period('D')
                            perc = execute(datas, k, None)
                            app = pd.DataFrame({'Kode Perusahaan': [k], 'Rata-rata Peningkatan (%)': [perc]})
                            result = result.append(app)
                        except:
                            pass
                    result = result.sort_values('Rata-rata Peningkatan (%)', ascending=False)
                    result = result[0:10]
                    result = result.set_index([pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])
                    st.markdown(f"<h4 style='text-align: center; '>Top 10 Stock's Growth Percentage on {sector}</h4>",
                                unsafe_allow_html=True)
                    st.table(result)


elif select == "Stock List":
    st.header("Kode Saham Indonesia")
    st.table(df)


