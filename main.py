import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import yfinance as yf
from pandas_datareader import data
import datetime

st.set_page_config(
    page_title="Saham Web App",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This Web App is made by Data Apes Team for TSDN 2022"
    }
)

st.markdown("<h2 style='text-align: center; '>Pemanfaatan Machine Learning dalam Memprediksi Nilai dan Valuasi Saham secara Teknikal dan Fundamental</h1>",
            unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; '>by Data Apes</h6>",
            unsafe_allow_html=True)
st.write("")
st.header('Latar Belakang')
st.markdown('<div style="text-align: justify;">Sebagai investor, tentunya tujuan utama yang diraih adalah keuntungan. Untuk itu, pentingnya untuk mengenali saham yang akan dibeli melalui berbagai analisis saham, salah duanya adalah analisis Teknikal dan Fundamental.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;">Analisis teknikal, yang biasanya digunakan untuk melakukan investasi jangka pendek, didasarkan pada data-data harga historis dimana prediksi untuk membeli atau menjual saham dilakukan dengan melihat grafik historis pergerakan saham.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;">Analisis fundamental, dimana umum digunakan untuk melakukan investasi jangka panjang, didasarkan oleh kondisi suatu perusahaan, kondisi ekonomi dan industri terkait menggunakan  indikator-indikator perusahaan yang tertera melalui laporan keuangan perusahaan seperti Price to Earning Ratio (P/E), ROE (Return to Equity), dan lain-lainnya.</div>', unsafe_allow_html=True)
st.markdown("""---""")
st.header('Pilih-Pilih:')
koloms1, koloms2, koloms3 = st.columns([1,1,1])
profil = koloms1.radio("Profil Investor:",('Jangka Pendek', 'Jangka Panjang'))
output = koloms2.radio("Output:",('Rekomendasi','Nilai Saham'))
koloms3.write("")

if output == "Rekomendasi":
    option = st.selectbox(
        'Pilih Sektor',
        ('Basic Materials', 'Communication Services', 'Consumer Cyclical', 'Consumer Defensive', 'Energy', 'Financial Services', 'Healthcare', 'Industrials', 'Real Estate', 'Technology', 'Utilities'))
elif output == "Nilai Saham":
    emiten = st.text_input('Kode Perusahaan (Contoh: AAPL)',"")
st.write("")

if output == "Nilai Saham":
    if emiten != "":
        symbol = yf.Ticker(emiten).info
        if symbol.get('longName') is not None:
            kol1, kol2 = st.columns([1,5])
            kol1.image(symbol.get('logo_url'))
            kol2.markdown(f"<h3 style='text-align: left; '>{symbol.get('longName')}</h3>", unsafe_allow_html=True)
            kol2.write(f"Country: {symbol.get('country')}")
            kol2.write(f"Currency: {symbol.get('currency')}")
            kol2.write(f"Sector: {symbol.get('sector')}")
            kol2.write(f"Website: {symbol.get('website')}")
            kol2.markdown(f"<div style='text-align: justify;'>{symbol.get('longBusinessSummary')}</div>", unsafe_allow_html=True)
            st.write("")

            if profil == "Jangka Pendek":
                st.markdown(f"<h4 style='text-align: center; '>Stock Price of {emiten}</h4>",
                            unsafe_allow_html=True)
                col1, col2 = st.columns([3,1])
                source = 'yahoo'
                start = col2.date_input(
                    "Start Date:",
                    datetime.date(2022, 1, 1))
                end = col2.date_input(
                    "End Date:")
                data = data.DataReader(emiten, start=start, end=end, data_source=source).reset_index()
                # st.write(data)
                def get_chart(data):
                    data["Close"] = round(data["Close"],2)
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
                chart = get_chart(data)
                col1.altair_chart(
                    (chart).interactive(),
                    use_container_width=True)
                st.write("")

#################

                st.markdown(f"<h4 style='text-align: center; '>Stock Price of {emiten} (2)</h4>",
                            unsafe_allow_html=True)
                cols1, cols2 = st.columns([3, 1])
                source = 'yahoo'
                start2 = cols2.date_input(
                    "Start Date: ",
                    datetime.date(2022, 10, 10))
                intv = cols2.radio("Interval:",('5m', '15m', '30m', '1h', '1d', '5d', '1wk'))
                data = yf.download(tickers=emiten, start = start2, interval = intv, rounding= True)
                data["Date"] = data.index.to_frame()

                chart = get_chart(data)
                cols1.altair_chart(
                    (chart).interactive(),
                    use_container_width=True)
                st.write("")

        else:
            st.markdown("<h6 style='text-align: center; '>Emiten tidak ditemukan</h6>", unsafe_allow_html=True)

st.write("")


