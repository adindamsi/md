import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class LoanModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_and_preprocess(self):
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)

        categorical_cols = ['person_gender', 'person_education', 'person_home_ownership',
                            'loan_intent', 'previous_loan_defaults_on_file']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]
        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_and_save(self, output_path="rf_model_optimized.pkl"):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, output_path, compress=3)
        print(f"Model saved to {output_path}")

if __name__ == "__main__":
    trainer = LoanModelTrainer("Dataset_A_loan.csv")
    trainer.train_and_save()