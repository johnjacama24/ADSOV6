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
            data["diccionario_genero"],
            data["diccionario_estado_civil"],
            data["dataframe_codificado_top5"],
        )

modelo, dicc_estado, dicc_genero, dicc_estado_civil, df_ref = cargar_modelo()

# ------------------------------
# Invertir los diccionarios para mostrar en el selectbox y mapear al código
# ------------------------------
inv_genero = {v: k for k, v in dicc_genero.items()}
inv_estado_civil = {v: k for k, v in dicc_estado_civil.items()}

# ------------------------------
# Interfaz de usuario
# ------------------------------
st.title("🧠 Predicción del Estado del Aprendiz")
st.markdown("Seleccione las opciones correspondientes y presione el botón para predecir.")

# Campos de entrada
edad = st.slider("Edad", 15, 60, 25)
reversiones = st.slider("Cantidad de Reversiones", 0, 10, 0)
quejas = st.slider("Cantidad de quejas", 0, 10, 0)
estrato = st.slider("Estrato", 0, 10, 0)

genero_opcion = st.selectbox("Género", list(dicc_genero.keys()))
estado_civil_opcion = st.selectbox("Estado Civil", list(dicc_estado_civil.keys()))

# ------------------------------
# Botón para predecir
# ------------------------------
if st.button("🔍 Realizar predicción"):
    try:
        fila = df_ref.drop(columns=["Estado Aprendiz"]).iloc[0].copy()

        fila["Edad"] = edad
        fila["Cantidad de quejas"] = quejas
        fila["Cantidad de Reversiones"] = reversiones
        fila["Género"] = dicc_genero[genero_opcion]
        fila["Estado Civil"] = dicc_estado_civil[estado_civil_opcion]

        entrada = pd.DataFrame([fila])

        pred_codificada = modelo.predict(entrada)[0]
        pred_original = dicc_estado.get(pred_codificada, "Desconocido")

        st.success(f"✅ Estado del aprendiz predicho: **{pred_original}**")
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
