import streamlit as st
import utils.funciones as fc

# configurar página
fc.config_page()

st.title('Predicción de ataques de apoplejía')

# menu
menu = st.sidebar.selectbox('Apoplejía', ['Portada', 'Datos', 'Modelos', 'Predecir', 'Conclusiones'])

if menu == 'Portada':
    fc.portada()

elif menu == 'Datos':
    fc.datos()

elif menu == 'Modelos':
    fc.modelos()

elif menu == 'Predecir':
    fc.predecir()

else:
    fc.conclusiones()