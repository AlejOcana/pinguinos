import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()
model_item = { "Decision Tree": DecisionTreeClassifier()}

for name, model in model_item.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, f'pinguinos/models/{name.lower().replace(" ", "_")}_model.pkl')
    print(f'Modelo {name} entrenado y guardado.')