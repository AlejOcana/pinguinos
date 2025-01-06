import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data():
    df = pd.read_csv("https://raw.githubusercontent.com/tnavarrete-iedib/bigdata-24-25/refs/heads/main/penguins_size.csv")
    df = df.dropna()

    X = df.drop(columns=["species"])
    y = df["species"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']),
            ('cat', OneHotEncoder(), ['island', 'sex'])
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor
