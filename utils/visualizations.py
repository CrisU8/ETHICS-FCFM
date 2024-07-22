from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download("punkt")
nltk.download("stopwords")
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_comment_lengths(
    df: pd.DataFrame, comment_column: str, title, output_file
) -> None:
    """
    Analyze the lengths of the comments in the DataFrame.
    Parameters:
    - df: DataFrame que contiene los comentarios.
    - comment_column: Nombre de la columna que contiene los comentarios.
    - title: Título de la visualización.
    - output_file: Nombre del archivo de salida para guardar la visualización.
    """
    # Leer el archivo CSV

    # Asegurarse de que la columna de comentarios existe
    if comment_column not in df.columns:
        raise ValueError(f"Columna '{comment_column}' no encontrada en el archivo CSV.")

    # Calcular la longitud de cada comentario
    df["comment_length"] = df[comment_column].apply(lambda x: len(str(x).split()))

    # Obtener estadísticas descriptivas
    length_stats = df["comment_length"].describe()
    print(length_stats)
    print(df["comment_length"])

    # Graficar la distribución de las longitudes de los comentarios
    plt.figure(figsize=(12, 6))
    plt.hist(df["comment_length"], bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Longitud")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, format="png")
    plt.show()


def extract_and_display_ngrams(text, n=2, top_k=10, output_file=None):
    """
    Extrae n-gramas del texto, cuenta sus frecuencias y muestra los n-gramas más frecuentes.

    Parameters:
    - text: Texto del que extraer los n-gramas.
    - n: Número de palabras en el n-grama (bigramas, trigramas, etc.).
    - top_k: Número de n-gramas más frecuentes a mostrar.
    - output_file: Nombre del archivo de salida para guardar la visualización.
    """
    # Tokenizar el texto
    tokens = nltk.word_tokenize(text.lower())

    # Filtrar stopwords
    stop_words = set(stopwords.words("spanish")) | set(
        ["hola", "chat", "ola", "bueno", "puse", "b", "bb", "f"]
    )
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Extraer n-gramas
    n_grams = ngrams(tokens, n)

    # Contar frecuencias de n-gramas
    n_gram_freq = Counter(n_grams)

    # Convertir a DataFrame para visualizar
    df = pd.DataFrame(n_gram_freq.items(), columns=["N-gram", "Frequency"])
    df = df.sort_values(by="Frequency", ascending=False).head(top_k)

    # Mostrar n-gramas y sus frecuencias
    print(df)

    # Graficar los n-gramas más frecuentes
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="Frequency",
        y=df["N-gram"].astype(str).apply(lambda x: " ".join(x)),
        data=df,
        palette="viridis",
    )
    plt.xlabel("Frequency")
    plt.ylabel(f"{n}-gram")
    plt.title(f"Top {top_k} {n}-grams")
    plt.tight_layout()
    plt.savefig(output_file, format="png")
    plt.show()


def analyze_ngrams_from_dataframe(df, comment_column, n=2, top_k=10, output_file=None):
    """
    Analiza n-gramas para cada comentario en un DataFrame y muestra los n-gramas más frecuentes.

    Parameters:
    - df: DataFrame que contiene los comentarios.
    - comment_column: Nombre de la columna que contiene los comentarios.
    - n: Número de palabras en el n-grama (bigramas, trigramas, etc.).
    - top_k: Número de n-gramas más frecuentes a mostrar.
    - output_file: Nombre del archivo de salida para guardar la visualización.
    """
    # Concatenar todos los comentarios en un solo texto
    all_comments = " ".join(df[comment_column].astype(str))

    # Extraer y mostrar n-gramas
    extract_and_display_ngrams(all_comments, n=n, top_k=top_k, output_file=output_file)
