import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def model_training(data):
    # Split data into features and target
    X = data.drop("cambio_postura", axis=1)
    y = data["cambio_postura"]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Obtener importancia de las características usando permutation_importance
    results = permutation_importance(model, X_test, y_test, scoring="accuracy")
    importance_df = pd.DataFrame(
        {"Feature": X_test.columns, "Importance": results.importances_mean}
    )

    # Ordenar por importancia
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print(f"Accuracy: {accuracy}")
    print(f"Report: {report}")
    print("Feature Importances (using Permutation Importance):")
    print(importance_df)

    return model, accuracy, report


if __name__ == "__main__":
    data = pd.read_csv("datos/processed/amanda_procesado.csv")
    data = data[data["df"] == 1]
    data = data[
        [
            "ind1",
            "group",
            "numero_mensajes_chat",
            "analisis",
            "razones",
            "contraargumentos",
            "punto_vista",
            "cambio_postura",
        ]
    ]
    model, accuracy, report = model_training(data)
    print(f"Accuracy: {accuracy}")
    print(f"Report: {report}")
