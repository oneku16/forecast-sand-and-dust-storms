import streamlit as st
import pandas as pd
import pickle
import plotly.express as px


bishkek_data = pd.read_csv("Datasets/Bishkek_data.csv")
data_Q = pd.read_csv("Datasets/pm2_data.csv")
data_Q.dropna(inplace=True)
pollutants_D = pd.read_csv("Datasets/grid-export.csv")

st.title('Open EMIT')

st.sidebar.header("Choose Page")
pages = st.sidebar.selectbox('Which page you want visit?',
                             ('Analyze', 'Predictions'))

if pages == 'Analyze':
    st.subheader("Air Quality in Bishkek")
    opt1 = st.selectbox("Visualize graph Based on", ('median', 'variance'))
    graph_1 = px.line(bishkek_data, x="Date", y=opt1, color='Specie')
    st.plotly_chart(graph_1)

    st.subheader("Quality of Air in Bishkek from 2019 to 2022")
    graph_2 = px.pie(data_Q, names='AQI Category')
    st.plotly_chart(graph_2)

    st.subheader('Concentration of pollutants in Bishkek Air')
    opt2 = st.selectbox("Which pollutant you want to visualize?", (
            'PM1(mcg/m³)',
            'PM10(mcg/m³)',
            'PM2.5(mcg/m³)',
            'NO(mcg/m³)',
            'NO2(mcg/m³)',
            'SO2(mcg/m³)',
            'Temperature(°C)'
    ))
    graph_3 = px.line(pollutants_D, x="Day", y=opt2)
    st.plotly_chart(graph_3)

if pages == 'Predictions':
    models = st.sidebar.selectbox('Which models you want to use?',
                                  ('Classification',))

    if models == 'Classification':
        option = st.selectbox(
            'You can use any of the given modules to predict the quality of air in Bishkek',
            ('KNN',))

        def user_input_features():
            year = st.sidebar.slider('Year', 2022, 2030, 2023)
            month = st.sidebar.slider('Month', 1, 12, 4)
            day = st.sidebar.slider('Day', 0, 6, 4)
            hour = st.sidebar.slider('Hour', 0, 23, 7)
            cons = st.sidebar.slider('NowCast', 0, 150, 46)
            aqi = st.sidebar.slider('AQI', 0, 600, 50)
            raw = st.sidebar.slider('Raq Conc.', 0, 150, 50)
            data = {
                    'Year': year,
                    'Month': month,
                    'Day': day, 'Hour': hour, 'NowCast Conc.': cons, 'AQI': aqi,
                    'Raw Conc.': raw}
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_features()
        st.subheader('User Input Values')
        st.write(df)

        if option == 'KNN':
            pickled_model = pickle.load(open('Classification/KNN', 'rb'))
            predictions = pickled_model.predict(df)
            if str(predictions[0]) == '0':
                air = 'The quality of predicted air is : Good'
            elif str(predictions[0]) == '1':
                air = 'The quality of predicted air is : Hazardous'
            elif str(predictions[0]) == '2':
                air = 'The quality of predicted air is : Moderate'
            elif str(predictions[0]) == '3':
                air = 'The quality of predicted air is : Unhealthy'
            elif str(predictions[0]) == '4':
                air = 'The quality of predicted air is : Unhealthy for Sensitive Groups'
            elif str(predictions[0]) == '5':
                air = 'The quality of predicted air is : Ver unhealthy'

            evaluation_classification = {'Accuracy': 0.81, 'Precision': 0.82, 'Recall': 0.91, 'F1-score': 0.89}

        st.subheader("Predictions")
        st.write(air)
        st.subheader("Performance of model on testing dataset")
        st.write(pd.DataFrame(evaluation_classification, index=[0]))
