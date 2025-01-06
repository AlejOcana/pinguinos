import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

joblib.dump(model, "pinguinos/models/logistic_regression_model.pkl")
print("Modelo de Regresión Logística entrenado y guardado.")