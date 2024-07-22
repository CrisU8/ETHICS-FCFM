import logging
import os

import coloredlogs
import numpy as np
import pandas as pd

# Configurar el logger
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="ethics.log",
)


def format_comments(messages):
    """
    Formatea los mensajes para incluir delimitadores de inicio y fin.

    Parameters:
    - messages: Lista de mensajes.

    Returns:
    - Cadena de texto formateada con delimitadores.
    """
    formatted = []
    for message in messages:
        formatted.append(f"[START_COMMENT] {message} [END_COMMENT]")
    return " ".join(formatted)


if __name__ == "__main__":
    # Definir la ruta de salida
    output_path = "datos/processed/amanda_procesado.csv"
    # Leer el archivo CSV
    logger.info("Leyendo el archivo CSV")
    amanda = pd.read_csv("datos/processed/amanda_all.csv")
    chat = pd.read_csv("datos/raw/amanda_chat.csv")
    logger.info(f"Las primeras filas del DataFrame son:\n{amanda.head()}")
    logger.info(f"Información general del DataFrame:\n{amanda['phase'].value_counts()}")
    # Columna cambio de postura
    # amanda["camioo_postura"] = amanda["5_señ"]

    # Agrupar el dataframe por columna user_id y agregar la columna comentarios_chat y cantidad de comentarios por chat
    amanda_group = (
        chat.groupby(["user_id", "team_id", "df"])
        .agg(
            chat_comments=("message", lambda msgs: format_comments(msgs)),
            numero_mensajes_chat=("message", "count"),
        )
        .reset_index()
    )
    # amanda.drop_duplicates(subset=["user_id", "df"], inplace=True)
    amanda = pd.merge(amanda, amanda_group, on=["user_id", "df"], how="left")
    amanda = amanda.drop_duplicates(subset=["user_id", "df"])

    # Columna cambio de postura, y estandarizacion de seleccion en fases
    amanda["cambio_postura"] = np.where(
        amanda["phase5_sel"] > 0,
        amanda["phase5_sel"] - amanda["phase1_sel"],
        amanda["phase3_sel"] - amanda["phase1_sel"],
    )
    amanda["ind1"] = amanda["phase1_sel"]
    amanda["group"] = np.where(
        amanda["phase5_sel"] > 0, amanda["phase3_sel"], amanda["phase2_sel"]
    )
    amanda["ind2"] = np.where(
        amanda["phase5_sel"] > 0, amanda["phase5_sel"], amanda["phase3_sel"]
    )
    amanda[["analisis", "razones", "contraargumentos", "punto_vista"]] = (
        np.random.randint(0, 2, size=(amanda.shape[0], 4))
    )
    amanda.to_csv(output_path, index=False)
    logger.info(f"Se ha guardado el DataFrame agrupado en {output_path}")
    logger.info(f"Las primeras filas del DataFrame agrupado son:{amanda.head()}")
    logger.info(f"Información general del DataFrame agrupado:\n{amanda.info()}")
    logger.info(
        f"Se ha guardado el DataFrame agrupado en {amanda.phase.value_counts()}"
    )
