import pandas as pd
import numpyas np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('dccuchile/bert-base-spanish-wwm-uncased')


if __name__ == "__main__":
    # Leer el archivo CSV
    path = "datos/processed/amanda_procesado.csv"
    df = pd.read_csv(path)
    # Mostrar las primeras filas del DataFrame
    logger.info(f"Las primeras filas del DataFrame son:\n{df.head()}")

    # Embeddings de los comentarios de justificacion y del chat
    df["comment_embedding"] = df["comment"].apply(lambda x: model.encode(x))
    df["chat_embedding"] = df["message"].apply(lambda x: model.encode(x))
