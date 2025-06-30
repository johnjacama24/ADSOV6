import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Cargar modelo, diccionario y dataframe de referencia
# ------------------------------
@st.cache_resource
def cargar_modelo():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        return (
            data["model"],
            data["label_encoder_mapping"],
            data["diccionario_desarrollar"],
            data["diccionario_recolectar"],
            data["diccionario_construir"],
            data["dataframe_codificado_top5"],
        )

modelo, dicc_estado, dicc_desarrollar, dicc_recolectar, dicc_construir, df_ref = cargar_modelo()

# ------------------------------
# Invertir los diccionarios para mostrar en el selectbox y mapear al código
# ------------------------------
inv_desarrollar = {v: k for k, v in dicc_desarrollar.items()}
inv_recolectar = {v: k for k, v in dicc_recolectar.items()}
inv_construir = {v: k for k, v in dicc_construir.items()}

# ------------------------------
# Interfaz de usuario
# ------------------------------
st.title("🧠 Predicción del Estado del Aprendiz")
st.markdown("Seleccione las opciones correspondientes y presione el botón para predecir.")

# Campos de entrada
edad = st.slider("Edad", 15, 60, 25)
quejas = st.slider("Cantidad de quejas", 0, 10, 0)

desarrollar_opcion = st.selectbox("Desarrollar procesos lógicos", list(dicc_desarrollar.keys()))
recolectar_opcion = st.selectbox("Recolectar información del software", list(dicc_recolectar.keys()))
construir_opcion = st.selectbox("Construir la base de datos", list(dicc_construir.keys()))

# ------------------------------
# Botón para predecir
# ------------------------------
if st.button("🔍 Realizar predicción"):
    try:
        fila = df_ref.drop(columns=["Estado Aprendiz"]).iloc[0].copy()

        fila["Edad"] = edad
        fila["Cantidad de quejas"] = quejas
        fila["DESARROLLAR PROCESOS LÓGICOS"] = dicc_desarrollar[desarrollar_opcion]
        fila["RECOLECTAR INFORMACIÓN DEL SOFTWARE"] = dicc_recolectar[recolectar_opcion]
        fila["CONSTRUIR LA BASE DE DATOS"] = dicc_construir[construir_opcion]

        entrada = pd.DataFrame([fila])

        pred_codificada = modelo.predict(entrada)[0]
        pred_original = dicc_estado.get(pred_codificada, "Desconocido")

        st.success(f"✅ Estado del aprendiz predicho: **{pred_original}**")
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
