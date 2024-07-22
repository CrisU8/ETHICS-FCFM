import logging
import os

import coloredlogs
import pandas as pd

# Configurar el logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
coloredlogs.install(level="DEBUG", logger=logger)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="ethics.log",
)


if __name__ == "__main__":
    # Definir la ruta de salida
    output_path = "datos/processed/amanda_all.csv"

    # Leer el archivo CSV
    amanda = pd.read_csv("datos/raw/amanda_all.csv")
    logger.info(f"Las primeras filas del DataFrame son:\n{amanda.head()}")
    # amanda.dropna(subset=["team_id"], inplace=True)
    logger.info(f"Información general del DataFrame:\n{amanda['phase'].value_counts()}")
    pivot = amanda.pivot_table(
        index=["user_id", "df"], columns="phase", values="sel", fill_value=0
    )
    pivot.columns = [f"phase{col}_sel" for col in pivot.columns]

    pivot_comments = amanda.pivot_table(
        index=["user_id", "df"], columns="phase", values="comment", aggfunc="first"
    )
    pivot_comments.columns = [f"phase{col}_comment" for col in pivot_comments.columns]
    logger.info(f"Se ha creado una tabla pivote:\n{pivot.head()}")
    amanda = pd.merge(amanda, pivot, on=["user_id", "df"], how="left")
    amanda = pd.merge(amanda, pivot_comments, on=["user_id", "df"], how="left")
    logger.info(f"Información general del DataFrame:\n{amanda.head()}")

    if not os.path.exists(output_path):
        # crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    amanda.to_csv(output_path, index=False)
