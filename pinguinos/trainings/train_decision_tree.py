import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

model.fit(X_train, y_train)

joblib.dump(model, "pinguinos/models/decision_tree_model.pkl")
print("Modelo Arbol de Decisión entrenado y guardado.")