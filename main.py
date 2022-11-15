import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import yfinance as yf

st.set_page_config(
    page_title="Saham Web App",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
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
            kol1, kol2 = st.columns([1,3])
            kol1.image(symbol.get('logo_url'))
            kol2.markdown(f"<h3 style='text-align: left; '>{symbol.get('longName')}</h3>", unsafe_allow_html=True)
            kol2.write(f"Country: {symbol.get('country')}")
            kol2.write(f"Currency: {symbol.get('currency')}")
            kol2.write(f"Sector: {symbol.get('sector')}")
            kol2.write(f"Website: {symbol.get('website')}")
            kol2.markdown(f"<div style='text-align: justify;'>{symbol.get('longBusinessSummary')}</div>", unsafe_allow_html=True)

        else:
            st.markdown("<h6 style='text-align: center; '>Emiten tidak ditemukan</h6>", unsafe_allow_html=True)


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
