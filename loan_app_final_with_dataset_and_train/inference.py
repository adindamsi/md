import pandas as pd
import joblib

class LoanModelInference:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_dict):
        df_input = pd.DataFrame([input_dict])
        return self.model.predict(df_input)[0]

if __name__ == "__main__":
    model_path = "rf_model_optimized.pkl"
    input_data = {
        'person_age': 28,
        'person_gender': 1,
        'person_education': 1,
        'person_income': 40000,
        'person_emp_exp': 3,
        'person_home_ownership': 2,
        'loan_amnt': 8000,
        'loan_intent': 2,
        'loan_int_rate': 13.0,
        'loan_percent_income': 0.2,
        'cb_person_cred_hist_length': 3,
        'credit_score': 680,
        'previous_loan_defaults_on_file': 0
    }

    predictor = LoanModelInference(model_path)
    result = predictor.predict(input_data)
    print(f"Hasil Prediksi Loan Status: {'Diterima' if result == 1 else 'Ditolak'}")