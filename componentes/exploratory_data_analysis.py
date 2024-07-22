import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.visualizations import *

# Configurar el logger
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="ethics.log",
)

if __name__ == "__main__":
    # Leer el archivo CSV
    path = "datos/processed/amanda_procesado.csv"

    df = pd.read_csv(path)

    # Mostrar las primeras filas del DataFrame
    logger.info(f"Las primeras filas del DataFrame son:\n{df.head()}")

    # Mostrar información general del DataFrame
    logger.info(f"Información general del DataFrame:\n{df.info()}")

    # Estadísticas descriptivas del DataFrame
    logger.info(f"Estadísticas descriptivas del DataFrame:\n{df.describe()}")

    # =============Visualizaciones================
    # Visualizar los latgos de la columna "comment"
    df.dropna(subset=["comment"], inplace=True)
    analyze_comment_lengths(
        df,
        "comment",
        "Distribución de las longitudes de los comentarios",
        "reportes/figures/comment_lengths.png",
    )
    # Visualizar la longitud de  los comentarios del chat
    chat = pd.read_csv("datos/raw/amanda_chat.csv")
    analyze_comment_lengths(
        chat,
        "message",
        "Distribución de las longitudes de los comentarios del chat",
        "reportes/figures/chat_lengths.png",
    )

    # Ver los n-gramas más frecuentes en los comentarios
    analyze_ngrams_from_dataframe(
        df, "comment", n=1, top_k=10, output_file="reportes/figures/unigrams.png"
    )
    analyze_ngrams_from_dataframe(
        df, "comment", n=2, top_k=10, output_file="reportes/figures/bigrams.png"
    )
    analyze_ngrams_from_dataframe(
        df, "comment", n=3, top_k=10, output_file="reportes/figures/trigrams.png"
    )
    analyze_ngrams_from_dataframe(
        df, "comment", n=4, top_k=10, output_file="reportes/figures/fourgrams.png"
    )

    # Ver los n-gramas más frecuentes en los mensajes del chat
    analyze_ngrams_from_dataframe(
        chat,
        "message",
        n=1,
        top_k=10,
        output_file="reportes/figures/chat_unigrams.png",
    )
    analyze_ngrams_from_dataframe(
        chat,
        "message",
        n=2,
        top_k=10,
        output_file="reportes/figures/chat_bigrams.png",
    )
    analyze_ngrams_from_dataframe(
        chat,
        "message",
        n=3,
        top_k=10,
        output_file="reportes/figures/chat_trigrams.png",
    )
    analyze_ngrams_from_dataframe(
        chat,
        "message",
        n=4,
        top_k=10,
        output_file="reportes/figures/chat_fourgrams.png",
    )
    df.cambio_postura.hist()
    plt.savefig("reportes/figures/cambio_postura.png")
    plt.title("Distribución de los cambios de postura")
    plt.xlabel("Cambio de postura")
    plt.ylabel("Frecuencia")
    plt.show()
