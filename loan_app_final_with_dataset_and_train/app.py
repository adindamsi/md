import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_model_optimized.pkl")

st.title("Loan Approval Prediction App")
st.subheader("Masukkan Data Pemohon:")

person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin", ["female", "male"])
person_education = st.selectbox("Pendidikan", ["High School", "Bachelor", "Master", "Other"])
person_income = st.number_input("Pendapatan Tahunan", min_value=0, value=50000)
person_emp_exp = st.number_input("Tahun Pengalaman Kerja", min_value=0, value=3)
person_home_ownership = st.selectbox("Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Jumlah Pinjaman", min_value=0, value=10000)
loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=12.5)
loan_percent_income = st.number_input("Rasio Pinjaman vs Pendapatan", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Durasi Histori Kredit (tahun)", min_value=0, value=3)
credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=650)
previous_loan_defaults_on_file = st.selectbox("Pernah Menunggak Sebelumnya?", ["No", "Yes"])

map_gender = {"female": 0, "male": 1}
map_edu = {"High School": 0, "Bachelor": 1, "Master": 2, "Other": 3}
map_home = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
map_loan_intent = {
    "EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2,
    "PERSONAL": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5
}
map_default = {"No": 0, "Yes": 1}

input_data = pd.DataFrame([{
    "person_age": person_age,
    "person_gender": map_gender[person_gender],
    "person_education": map_edu[person_education],
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": map_home[person_home_ownership],
    "loan_amnt": loan_amnt,
    "loan_intent": map_loan_intent[loan_intent],
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": map_default[previous_loan_defaults_on_file]
}])

if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    status = "DITERIMA ✅" if pred == 1 else "DITOLAK ❌"
    st.subheader(f"Hasil Prediksi Pinjaman: {status}")