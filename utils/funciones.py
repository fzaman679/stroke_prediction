import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import streamlit.components.v1 as components
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot
import sys, os
import utils.textos as tx
import base64
import pickle

# configuración página
def config_page():
    st.set_page_config(
        page_title = 'Apoplejía',
        layout = 'wide'
    )

    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# cache
st.cache(suppress_st_warning = True)

def mostrar(path, width):
    img = Image.open(path)
    st.image(img, width=width)

def cargar_datos(path):
    df = pd.read_csv(path)
    return df

def generar(orden, df):
    sns.barplot(x=orden,y='Model',data=df,color='b')
    plt.title('Model Compare Graphic');
    plt.savefig("utils/image/comp.jpg")

def portada():

    mostrar('utils/image/apo.jpg', 1300)

    with st.expander('¿Qué es la apoplejía?'):
        st.write(tx.pres)
        st.write(tx.muertes)
    
    with st.expander('Datasets:'):
        st.write(tx.link)

def datos():
    bt_var = st.sidebar.checkbox('¿Desea ver la distribución de las variables continuas?')
    if bt_var == False:
        st.markdown('##')
        st.markdown('<p class="big-font">Información</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)          
        with col1:
            mostrar('utils/image\info.png', 500)
        with col2:
            st.write(tx.atributos)
        
        st.markdown('##')
        st.markdown('<p class="big-font">Descripción</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)          
        with col1:
            mostrar('utils/image\desc.png', 500)
        with col2:
            st.write(tx.desc)

        st.markdown('##')
        st.markdown('<p class="big-font">Valores nulos</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)          
        with col1:
            mostrar('utils/image\sum.png', 300)
        with col2:
            st.write(tx.arreglar)

        st.markdown('##')
        st.markdown('##')
        
        bt_corr = st.checkbox('¿Desea ver la matriz de correlación?')
        if bt_corr:    
            mostrar('utils/image/corr.jpg', 1200)
    
    else:
        st.markdown('##')
        st.markdown('<p class="big-font">Distribución</p>', unsafe_allow_html=True)
        option = st.selectbox('Diferenciar entre sanos y afectados', ('No', 'Sí'))
        if option == 'No':
            mostrar('utils/image/var_ent.jpg', 1200)
        else:
            mostrar('utils/image/var_sep.jpg', 1200)
    
def modelos():
    st.markdown('##')
    st.markdown('<p class="big-font">Clasificación</p>', unsafe_allow_html=True)

    xgb = st.sidebar.checkbox('XGBClassifier')
    if xgb:

        col1, col2 = st.columns(2) 
        with col1:
            mostrar('utils/image/report.png', 600)
        with col2:
            st.write(tx.report)

        st.markdown('##')

        col1, col2 = st.columns(2) 
        with col1:
            mostrar('utils/image/mat.jpg', 700)
        with col2:
            st.markdown('##')
            st.markdown('##')
            st.markdown('##')
            st.write(tx.mat)
        
        st.markdown('##')

        col1, col2 = st.columns(2) 
        with col1:
            mostrar('utils/image/roc.jpg', 700)
        with col2:
            st.markdown('##')
            st.markdown('##')
            st.markdown('##')
            st.write(tx.roc)      
    
    else:
        df = cargar_datos('data\comparaciones.csv')
        orden = st.selectbox('Seleccione como ordenar los valores:', ('Accuracy', 'K-Fold Mean Accuracy', 'Std.Deviation'))

        df.sort_values(by=orden, ascending=False, inplace=True)
        st.write(df)

        generar(orden, df)
        st.markdown('##')
        mostrar('utils/image/comp.jpg', 1000)

def predecir():

    st.markdown('##')
    name = 'modelo'
    with open(name, 'rb') as file:  
        model = pickle.load(file)

    lista = []

    genre = st.radio("¿Cuál es su género?", ('Hombre', 'Mujer', 'Otro'))
    if genre == 'Hombre':
        lista.append(0)
    elif genre == 'Mujer':
        lista.append(1)
    else:
        lista.append(2)
    st.markdown('##')

    age = st.number_input('Inserte edad', min_value=0, max_value=150, step=1)
    lista.append(age)
    st.markdown('##')

    tension = st.radio("¿Padece hipertensión?", ('Sí', 'No'))
    if tension == 'Sí':
        lista.append(1)
    else:
        lista.append(0)
    st.markdown('##')

    cardio = st.radio("¿Padece alguna enfermedad cardiovascular?", ('Sí', 'No'))
    if cardio == 'Sí':
        lista.append(1)
    else:
        lista.append(0)
    st.markdown('##')

    casado = st.radio("¿Alguna vez se ha casdado?", ('Sí', 'No'))
    if casado == 'Sí':
        lista.append(1)
    else:
        lista.append(0)
    st.markdown('##')

    work = st.radio("¿Qué tipo de trabajo tiene?", ('Nunca he trabajado', 'Demasiado joven para trabajar', 'Empresa pública', 'Autónomo', 'Empresa privada'))
    if work == 'Nunca he trabajado':
        lista.append(0)
    elif work == 'Demasiado joven para trabajar':
        lista.append(1)
    elif work == 'Empresa pública':
        lista.append(4)
    elif work == 'Autónomo':
        lista.append(3)
    else:
        lista.append(2)
    st.markdown('##')

    casa = st.radio("¿Qué tipo de residencia tiene?", ('Rural', 'Urbana'))
    if casa == 'Urbana':
        lista.append(1)
    else:
        lista.append(0)
    st.markdown('##')

    glucosa = st.number_input("¿Cuál es su nivel medio de glucosa en sangre?", min_value=0.0, max_value=500.0, step=0.1)
    lista.append(glucosa)
    st.markdown('##')

    imc = st.number_input("¿Cuál es su índice de masa corporal?", min_value=0.0, max_value=100.0, step=0.1)
    lista.append(imc)
    st.markdown('##')

    fuma = st.radio("¿Se considera fumador?", ('He fumado', 'Nunca he fumado', 'Fumo', 'Otro'))
    if fuma == 'He fumado':
        lista.append(1)
    elif genre == 'Nunca he fumado':
        lista.append(0)
    elif genre == 'Fumo':
        lista.append(2)
    else:
        lista.append(1)

    st.markdown('##')

    path = 'data/columnas'
    with open(path, "rb") as f:
        columnas = pickle.load(f)

    lista = [lista]
    X_test = pd.DataFrame(lista, columns=columnas)

    path = 'model/modelo'
    with open(path, 'rb') as file:  
        model = pickle.load(file)

    y_pred = model.predict(X_test)
    st.markdown('##')

    if y_pred[0] == 0:
        st.markdown("<h1 style='text-align: center;'>Tranquilo, no sufrirás apoplejía</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: red;'>Cuidado, puedes sufrir un ataque de apoplejía!</h1>", unsafe_allow_html=True)
