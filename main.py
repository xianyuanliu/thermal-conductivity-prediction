import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "Data-Keff-Num.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Train R^2: {r2_score(y_train, model.predict(X_train)):.4f}")
    print(f"Test  R^2: {r2_score(y_test, model.predict(X_test)):.4f}")
    print("Coefficients:", dict(zip(FEATURES, model.coef_)))
    print(f"Intercept: {model.intercept_:.6g}")


if __name__ == "__main__":
    main()
