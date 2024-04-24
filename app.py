import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
# Cargar el DataFrame

modelrecomendacion = joblib.load('modelo_app')
# Hacer predicciones

# Realizar la predicción con los parámetros ingresados
def llamar_modelo(ultima_marca_comprada, ultima_cantidad_comprada, precio_marca_1, precio_marca_2, precio_marca_3, precio_marca_4, precio_marca_5, promo_marca_1, promo_marca_2, promo_marca_3, promo_marca_4, promo_marca_5, genero, estado_civil, edad, nivel_educacion, ingreso_anual, ocupacion, dias_acumulados):
    user_input = [ultima_marca_comprada, ultima_cantidad_comprada, precio_marca_1, precio_marca_2, precio_marca_3, precio_marca_4, precio_marca_5, promo_marca_1, promo_marca_2, promo_marca_3, promo_marca_4, promo_marca_5, genero, estado_civil, edad, nivel_educacion, ingreso_anual, ocupacion, dias_acumulados]
    example_prediction = modelrecomendacion.predict([user_input])
    st.write(f'Predicción de ejemplo: Producto Recomendado{example_prediction}')
# Definir la aplicación Streamlit
st.title('Modelo de Recomendación')
st.write(f'Accuracy: 0.7349726775956285')

# Agregar widgets para ingresar los parámetros de ejemplo
with st.form('datos_modelo'):
    ultima_marca_comprada = st.number_input("Ultima marca comprada", min_value=0, max_value=5)
    ultima_cantidad_comprada = st.number_input("Ultima cantidad comprada", min_value=0, max_value=100)
    precio_marca_1 = st.number_input("Precio marca 1", min_value=0.00, max_value=10.00)
    precio_marca_2 = st.number_input('Precio marca 2', min_value=0.00, max_value=10.00)
    precio_marca_3 = st.number_input('Precio marca 3', min_value=0.00, max_value=10.00)
    precio_marca_4 = st.number_input('Precio marca 4', min_value=0.00, max_value=10.00)
    precio_marca_5 = st.number_input('Precio marca 5', min_value=0.00, max_value=10.00)
    promo_marca_1 = st.radio('Promo marca 1', [0,1], key=None)
    promo_marca_2 = st.radio('Promo marca 2', [0,1], key=None)
    promo_marca_3 = st.radio('Promo marca 3', [0,1], key=None)
    promo_marca_4 = st.radio('Promo marca 4', [0,1], key=None)
    promo_marca_5 = st.radio('Promo marca 5', [0,1], key=None)
    genero = st.radio('Genero', [0,1], key=None)
    estado_civil = st.radio('Estado civil', [0,1], key=None)
    edad = st.number_input('Edad', min_value=18, max_value=100)
    nivel_educacion = st.radio('Nivel de educacion', [0,1,2,3], key=None)
    ingreso_anual = st.number_input('Ingreso anual', min_value=50000, max_value=300000)
    ocupacion = st.radio('Ocupacion', [0,1,2], key=None)
    dias_acumulados = st.number_input('Dias acumulados sin compra', min_value=0, max_value=500)
    submitted = st.form_submit_button("Enviar",on_click=llamar_modelo(ultima_marca_comprada, ultima_cantidad_comprada, precio_marca_1, precio_marca_2, precio_marca_3, precio_marca_4, precio_marca_5, promo_marca_1, promo_marca_2, promo_marca_3, promo_marca_4, promo_marca_5, genero, estado_civil, edad, nivel_educacion, ingreso_anual, ocupacion, dias_acumulados))