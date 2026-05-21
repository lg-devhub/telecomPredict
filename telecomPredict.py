import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("ChurnData.csv")

##X = df.drop("churn", axis=1)
X = df[[
    "age",
    "income",
    "tenure"
]]

y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Acurácia:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))


##UTILIZANDO ENV COM PY 3.11 POR CONTA DAS BIBLIOTECAS SCIKIT-LEARN E PANDAS. EXTENSÃO .ipynb ESTAVA DANDO CONFLITO DE EXTENSÕES POR ISSO EFETUEI DIRETO EM .py




import joblib

joblib.dump(model, "modelo_churn.pkl")
joblib.dump(scaler, "scaler.pkl")


