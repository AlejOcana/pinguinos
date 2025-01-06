import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

model.fit(X_train, y_train)

joblib.dump(model, "pinguinos/models/svm_model.pkl")
print("Modelo SVM entrenado y guardado.")