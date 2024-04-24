import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el DataFrame
data_compras= pd.read_csv("compras_data.csv", sep=",")
data_compras_dias = data_compras.copy()
data_compras_dias['dias_desde_ultima_visita'] = data_compras_dias.groupby('id')['dia_visita'].diff().fillna(0).astype(int)

# Reiniciar la acumulación por cada usuario (id)
current_id = None
for index, row in data_compras_dias.iterrows():
    if current_id != row['id']:
        current_id = row['id']
        dias_acumulados = 0
    else:
        dias_acumulados += row['dias_desde_ultima_visita']

    data_compras_dias.at[index, 'dias_acumulados'] = dias_acumulados

    if row['incidencia_compra'] == 1:
        dias_acumulados = 0

data_compras_concretadas = data_compras_dias[data_compras_dias['incidencia_compra'] == 1]

# Preparar los datos
X_rec = data_compras_concretadas.drop(columns=['id','incidencia_compra','dia_visita','id_marca','cantidad','dias_desde_ultima_visita','tamanio_ciudad'])
Y_rec = data_compras_concretadas['id_marca']
X_rec_train, X_rec_test, Y_rec_train, Y_rec_test = train_test_split(X_rec,Y_rec, test_size = 0.05)

# Entrenar el modelo
modelrecomendacion = DecisionTreeClassifier()
modelrecomendacion.fit(X_rec_train, Y_rec_train)

# Hacer predicciones
recomendaciones = modelrecomendacion.predict(X_rec_test)
accuracy = accuracy_score(Y_rec_test, recomendaciones)
# Realizar la predicción con los parámetros ingresados
def llamar_modelo(ultima_marca_comprada, ultima_cantidad_comprada, precio_marca_1, precio_marca_2, precio_marca_3, precio_marca_4, precio_marca_5, promo_marca_1, promo_marca_2, promo_marca_3, promo_marca_4, promo_marca_5, genero, estado_civil, edad, nivel_educacion, ingreso_anual, ocupacion, dias_acumulados):
    user_input = [ultima_marca_comprada, ultima_cantidad_comprada, precio_marca_1, precio_marca_2, precio_marca_3, precio_marca_4, precio_marca_5, promo_marca_1, promo_marca_2, promo_marca_3, promo_marca_4, promo_marca_5, genero, estado_civil, edad, nivel_educacion, ingreso_anual, ocupacion, dias_acumulados]
    example_prediction = modelrecomendacion.predict([user_input])
    st.write(f'Predicción de ejemplo: Producto {example_prediction}')
# Definir la aplicación Streamlit
st.title('Modelo de Recomendación')
st.write(f'Accuracy: {accuracy}')

# Agregar widgets para ingresar los parámetros de ejemplo
with st.form('datos_modelo'):
    ultima_marca_comprada = st.number_input("Ultima marca comprada", min_value=0, max_value=5)
    ultima_cantidad_comprada = st.number_input("Ultima cantidad comprada", min_value=0, max_value=100)
    precio_marca_1 = st.number_input("precio_marca_1", min_value=0.00, max_value=10.00)
    precio_marca_2 = st.number_input('precio_marca_2', min_value=0.00, max_value=10.00)
    precio_marca_3 = st.number_input('precio_marca_3', min_value=0.00, max_value=10.00)
    precio_marca_4 = st.number_input('precio_marca_4', min_value=0.00, max_value=10.00)
    precio_marca_5 = st.number_input('precio_marca_5', min_value=0.00, max_value=10.00)
    promo_marca_1 = st.radio('promo_marca_1', [0,1], key=None)
    promo_marca_2 = st.radio('promo_marca_2', [0,1], key=None)
    promo_marca_3 = st.radio('promo_marca_3', [0,1], key=None)
    promo_marca_4 = st.radio('promo_marca_4', [0,1], key=None)
    promo_marca_5 = st.radio('promo_marca_5', [0,1], key=None)
    genero = st.radio('genero', [0,1], key=None)
    estado_civil = st.radio('estado_civil', [0,1], key=None)
    edad = st.number_input('edad', min_value=20, max_value=100)
    nivel_educacion = st.radio('nivel_educacion', [0,1,2,3], key=None)
    ingreso_anual = st.number_input('ingreso_anual', min_value=50000, max_value=300000)
    ocupacion = st.radio('ocupacion', [0,1,2], key=None)
    dias_acumulados = st.number_input('dias_acumulados', min_value=0, max_value=500)
    submitted = st.form_submit_button("Enviar",on_click=llamar_modelo(ultima_marca_comprada, ultima_cantidad_comprada, precio_marca_1, precio_marca_2, precio_marca_3, precio_marca_4, precio_marca_5, promo_marca_1, promo_marca_2, promo_marca_3, promo_marca_4, promo_marca_5, genero, estado_civil, edad, nivel_educacion, ingreso_anual, ocupacion, dias_acumulados))