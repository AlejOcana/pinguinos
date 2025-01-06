import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()
model_item = {"KNN": KNeighborsClassifier()}

for name, model in model_item.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, f'pinguinos/models/{name.lower().replace(" ", "_")}_model.pkl')
    print(f'Modelo {name} entrenado y guardado.')