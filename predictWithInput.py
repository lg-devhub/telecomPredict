import streamlit as st
import joblib
import numpy as np

# Carregar modelo e scaler
model = joblib.load("modelo_churn.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Predição de Churn")

# Inputs
idade = st.number_input("Idade", min_value=0)
salario = st.number_input("Renda", min_value=0.0)
tempo_cliente = st.number_input("Tempo como cliente (meses)", min_value=0)

if st.button("Prever"):

    # Dados no MESMO formato do treino
    dados = np.array([[idade, salario, tempo_cliente]])

    # Aplicar scaler
    dados_scaled = scaler.transform(dados)

    # Predição
    pred = model.predict(dados_scaled)
    prob = model.predict_proba(dados_scaled)[0][1]

    st.write(f"Probabilidade de churn: {prob:.2%}")

    if pred[0] == 1:
        st.error("Cliente provavelmente vai cancelar.")
    else:
        st.success("Cliente provavelmente NÃO vai cancelar.")