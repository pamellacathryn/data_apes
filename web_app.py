import streamlit as st
import altair as alt
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def find_sector_emiten(emiten):
    sector = data_sector.loc[data_sector[0] == emiten, 2].values[0]
    return sector


def predict(index, emiten, orde, dataset):
    dataset1 = dataset.loc[dataset['Emiten'] == emiten, ['Peningkatan', col_names[index]]]
    dataset1.columns = [1, 2]
    [x, y] = [dataset1.loc[:, 1], dataset1.loc[:, 2]]
    mymodel = np.poly1d(np.polyfit(x, y, orde))
    test = mymodel(2022)
    score = r2_score(y, mymodel(x))

    return [test, score]


def execute(sector, dataset):
    test_pred = pd.DataFrame({col: [] for col in col_names[:-1]})
    test_acc = pd.DataFrame({col: [] for col in col_names[:-1]})
    orde = 3

    for i in emiten_sector.loc[emiten_sector[2] == sector, 0]:
        pred_list = [i, sector, 2022]
        acc_list = [i, sector, 2022]
        try:
            for j in range(3, len(col_names) - 1):
                test = predict(j, i, orde, dataset)
                pred_list.append(test[0])
                acc_list.append(test[1])

            df_app1 = pd.DataFrame({c: [v] for c, v in zip(col_names[:-1], pred_list)})
            test_pred = pd.concat([test_pred, df_app1])
            df_app2 = pd.DataFrame({c: [v] for c, v in zip(col_names[:-1], acc_list)})
            test_acc = pd.concat([test_acc, df_app2])

        except:
            pass

    return (test_pred, test_acc)


def gen_sector(sector):
    X = datazz.iloc[:, 3:11].values
    Y = datazz.iloc[:, 11].values

    X_train, _, _, _ = train_test_split(X, Y, test_size=0.2, random_state=20000)

    sc = StandardScaler()
    sc.fit(X_train)

    col_sc = col_names[3:11]

    arr_sc = sc.transform(datazz.iloc[:, 3:11])
    arr_sc_ = np.array([arr_sc[:, i] for i in range(8)])

    data_sc = pd.DataFrame({k: v for k, v in zip(col_sc, arr_sc_)})
    data_sc1 = datazz.iloc[:, 0:3]

    perc_sc = datazz.iloc[:, -1]

    datasetgen = pd.concat([data_sc1, data_sc, perc_sc], axis=1)

    data_pred, data_acc = execute(sector, datasetgen)

    return (data_pred, datasetgen)


def fundamental_rekom(sector):
    data_pred, datasetgen = gen_sector(sector)
    pickled_model = pickle.load(open(f'Regressor {sector}.pkl', 'rb'))

    inc_pred = pd.DataFrame({'Emiten': [], 'Percentage Increase (%)': []})
    emit = data_pred['Emiten'].unique()

    rasio_cols = data_pred.columns[3:]

    for e in emit:
        arr_pred = data_pred.loc[data_pred['Emiten'] == e, rasio_cols].values.reshape(1, -1)
        preds = pickled_model.predict(arr_pred)

        pred_app = pd.DataFrame({'Emiten': [e], 'Percentage Increase (%)': preds[0]})
        inc_pred = pd.concat([inc_pred, pred_app])

    inc_pred = inc_pred.sort_values('Percentage Increase (%)', ascending=False)

    X_test_ = datasetgen.iloc[:, 3:11]
    Y_test_ = datasetgen.iloc[:, 11]
    Y_pred_ = pickled_model.predict(X_test_)

    skor = np.abs(mean_squared_error(Y_pred_, Y_test_)) ** 0.5

    return (inc_pred, skor)

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

df = pd.read_table("Saham per Sektor(3).csv",delimiter=",")
data_sector = pd.read_csv("Saham per Sektor Tanpa JK.csv", header=None, delimiter=",", skiprows=1)
emiten_sector = data_sector.iloc[:, [0,2]]

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
    st.markdown("<h6 style='text-align: center; '>by Data Apes</h6>",
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
                if profil == "Long Term":
                    datazz = pd.read_csv(f"{sector}.csv", delimiter=",").drop('Unnamed: 0', axis=1)
                    col_names = datazz.columns
                    sector = find_sector_emiten(emiten)
                    top_10_rekomendasi, skor = fundamental_rekom(sector)
                    tablezz = top_10_rekomendasi.loc[top_10_rekomendasi["Emiten"] == emiten,:]
                    hide_table_row_index = """
                                <style>
                                thead tr th:first-child {display:none}
                                tbody th {display:none}
                                </style>
                                """
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(tablezz)
                    meannn = tablezz["Percentage Increase (%)"].iloc[0]
                    st.warning(f'The Percentage Increase of {tablezz["Emiten"].iloc[0]} may vary between {round(meannn-skor, 2)}% and {round(meannn+skor, 2)}%', icon="💡")
                    st.write("")
                    rasio = st.selectbox(
                            'Choose ratio:',
                            ('PB', 'ROA', 'ROE', 'EPS', 'PER', 'DER', 'DAR', 'Cash Flow', 'Percentage Increase (%)'))
                    st.markdown(f"<h4 style='text-align: center; '>Visualization of the {rasio} of {emiten} to the Average {rasio} of the {sector} Sector</h4>", unsafe_allow_html=True)
                    yearly_sektor_mean = datazz.groupby('Peningkatan').mean()[rasio]
                    yearly_emiten = datazz.loc[datazz['Emiten'] == emiten, :].groupby('Peningkatan').mean()[rasio]
                    gabungin = pd.concat([yearly_sektor_mean, yearly_emiten], axis=1)
                    gabungin.columns = [rasio+"_Sector",rasio]
                    st.line_chart(gabungin)


                elif profil == "Short Term":
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
                        datumz = yf.download(k, start)

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

            elif profil == "Long Term":
                datazz = pd.read_csv(f"{sector}.csv", delimiter=",").drop('Unnamed: 0', axis=1)
                col_names = datazz.columns
                top_10_rekomendasi, skor = fundamental_rekom(sector)
                if sector == "Technology":
                    par = 8
                else:
                    par = 10
                st.markdown(f"<h4 style='text-align: center; '>Top {par} Stock's Growth Percentage on {sector}</h4>",
                            unsafe_allow_html=True)
                top_10_rekomendasi = top_10_rekomendasi[0:10]
                top_10_rekomendasi.index = [i for i in range(1,len(top_10_rekomendasi)+1)]
                st.table(top_10_rekomendasi)
                st.warning(f'The Percentage Increase may vary between ±{round(skor,2)}% of the value shown above', icon="💡")
                st.write("")
                kyolumn1, kyolumn2 = st.columns([1,1])
                list_em = []
                for i in top_10_rekomendasi["Emiten"].to_list():
                    list_em.append(i)
                emiten = kyolumn1.selectbox(
                    'Choose emiten:',
                    (list_em))
                rasio = kyolumn2.selectbox(
                    'Choose ratio:',
                    ('PB', 'ROA', 'ROE', 'EPS', 'PER', 'DER', 'DAR', 'Cash Flow', 'Percentage Increase (%)'))
                st.markdown(
                    f"<h4 style='text-align: center; '>Visualization of the {rasio} of {emiten} to the Average {rasio} of the {sector} Sector</h4>",
                    unsafe_allow_html=True)
                yearly_sektor_mean = datazz.groupby('Peningkatan').mean()[rasio]
                yearly_emiten = datazz.loc[datazz['Emiten'] == emiten, :].groupby('Peningkatan').mean()[rasio]
                gabungin = pd.concat([yearly_sektor_mean, yearly_emiten], axis=1)
                gabungin.columns = [rasio + "_Sector", rasio]
                st.line_chart(gabungin)

elif select == "Stock List":
    st.header("Indonesia Stock Indexes")
    df_nonjk = df.copy()
    for i in range(len(df_nonjk)):
        string_1 = df_nonjk.iloc[i]["Kode"]
        string_2 = ".JK"
        if string_2 in string_1:
            baru = string_1.replace(string_2, '')
        df_nonjk.iloc[i]["Kode"] = baru
    df_nonjk.index = [i for i in range(1,len(df_nonjk)+1)]
    st.table(df_nonjk)


