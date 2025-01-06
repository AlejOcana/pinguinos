import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

model.fit(X_train, y_train)

joblib.dump(model, "pinguinos/models/knn_model.pkl")
print("Modelo KNN entrenado y guardado.")