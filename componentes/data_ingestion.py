import logging
import os

import coloredlogs
import logger
import pandas as pd

# Configurar el logger
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="ethics.log",
)


def merge_csv_files(folder_path, output_file):
    """
    Combina todos los archivos CSV en una carpeta específica en un solo DataFrame y lo guarda en un archivo CSV.

    Parameters:
    - folder_path: ruta a la carpeta que contiene los archivos CSV.
    - output_file: nombre del archivo de salida (incluye la extensión .csv).
    """
    # Lista para almacenar cada DataFrame
    dataframes = []

    # Recorrer todos los archivos en la carpeta
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, delimiter=";")
            dataframes.append(df)

    # Combinar todos los DataFrames en uno solo
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Guardar el DataFrame combinado en un archivo CSV
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Archivos combinados guardados en {output_file}")


if __name__ == "__main__":
    # Juntar archivos csv en un solo dataframe
    if not os.path.exists("datos/raw"):
        os.makedirs("datos/raw")

    merge_csv_files("datos/amanda/", "datos/raw/amanda_all.csv")
    logger.info("Procesando chats de Amanda...")
    merge_csv_files("datos/amanda/chat", "datos/raw/amanda_chat.csv")
