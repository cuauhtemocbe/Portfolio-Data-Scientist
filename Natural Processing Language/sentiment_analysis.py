# -*- coding: utf-8 -*-
"""Sentiment Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s6Fcr1_rWGOzebXfR2Pcl6vDmgKtdtuC

# Load data

Leyendo el csv
"""

import pandas as pd

df = pd.read_csv("/content/flipkart_review_data_2022_02.csv")
df = df.drop(columns="Unnamed: 0")

# Total de elementos
df.shape

df.head(3)

"""A primera vista, la variable objetivo podría ser la columna "averageRating" o "reviewTitle"
"""

# Conteo inicial de la variable 'target'
df.groupby("averageRating")["averageRating"].count()

# Revisando los datos, es necesario organizarlo en positive/negative
# y disminuir la cardinalidad de la tabla
df.groupby("reviewTitle")["reviewTitle"].count().sort_values(ascending=False).head(10)

# Podríamos filtrar los comentarios por likes
# dejando los más relevantes, o eliminado aquellos
# con mayor dislikes
df.describe()

# Eliminaré el 5% de comentarios con más dislikes
q_05_dislikes = df["reviewDislikes"].quantile(0.99)
q_05_dislikes

df = df[df.reviewDislikes < q_05_dislikes]

# Después del filtrado nos quedamos con 325 comentarios
df.shape

# Creando dos etiquetas: Bueno y Malo

# [1] Obteniendo valores únicos

unique_titles = df["reviewTitle"].unique()
unique_titles

# Etiquetado manual de la información
new_categories_dict = {
    "good": ['Excellent', 'Really Nice', 'Super!', 'Just wow!',
       'Highly recommended', 'Great product', 'Worth the money',
       'Good quality product', 'Nice', 'Decent product',
       'Perfect product!', 'Classy product', 'Terrific', 'Good choice',
       'Fabulous!', 'Worth every penny', 'Wonderful', 'Simply awesome',
       'Very Good', 'Terrific purchase', 'Brilliant',
       'Mind-blowing purchase', 'Must buy!', 'Pretty good', 'Awesome',
       'Best in the market!', 'Value-for-money', 'Fair', 'Nice product',
       'Delightful', 'Good', 'Does the job'],

    "bad": ['Slightly disappointed', 'Hated it!', 'Just okay',
            'Could be way better', 'Useless product', 'Unsatisfactory',
            'Very poor', 'Moderate', 'Not good',
            'Utterly Disappointed', 'Worst experience ever!',
            'Expected a better product', 'Not recommended at all',
            'Waste of money!', 'Absolute rubbish!',
            'Did not meet expectations', "Hated it!",
            "Not good", "Useless product"]
}

def relabel_target(title: pd.Series):
  """Returns a new category according to the new labes dictionary"""
  new_category = "not found"

  for category in new_categories_dict.keys():
    if title in new_categories_dict[category]:
      new_category = category
      break

  return new_category

df["sentiment"] = df["reviewTitle"].apply(relabel_target)

# Nueva distribución de categorias
# Lo primero que podemos observar es que se encuentra
# Imbalanceado
df.groupby("sentiment")["sentiment"].count()

"""# Train, Test"""

from sklearn.model_selection import train_test_split

# Normalmente se usa un 80-20 o 70-30 para
# para el modelo, pero ya que son muy pocos datos,
# se utilizará un muestra pequena para testear los datos
train, test = train_test_split(df, test_size=0.2,
                               random_state=0, stratify=df["sentiment"])

train.groupby("sentiment")["sentiment"].count()

round(train.groupby("sentiment")["sentiment"].count() / train.shape[0] * 100, 2)

test.groupby("sentiment")["sentiment"].count()

round(test.groupby("sentiment")["sentiment"].count() / test.shape[0] * 100, 2)

"""Para tratar datos no balanceados, podemos:
- Realizar un sub-muestreo de la data que tiene mayor cantidad de datos
- Realizar un sobre-muestreo de la etiqueta con menor cantidad de datos
- Usar K-fold cross validation
- Ensamblado de diferentes modelos, realizando un submuestreo de la categoría con mayor cantidad de datos
y siempre usando la categoría con menos datos

Por el tiempo disponible optare por K-fold cross validation que se integra dentro GridSearch.

# Preprocess (data cleaning)

La columna reviewDescription será utilizada como feature para la construcción del modelo, el preprocesamiento incluye:

- Eliminar signos de puntuación
- Convertir a minúsculas
- Remover palabras más comunes (stopwords)
- Lematización: llevar a la raiz
- Remover palabras menores a 2 letras
- Vectorizar la información
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargando librería stopwords
nltk.download('stopwords')
# Descargando librería para lematización
nltk.download('wordnet')

# Obteniendo stopwords para idioma inglés
STOPWORDS = nltk.corpus.stopwords.words('english')

# Instalando librería para eliminar emojis
!pip install emoji

import emoji

LEMMATIZER = WordNetLemmatizer()

def remove_stopwords(text: str) -> str:
  text_processed = " ".join(word for word in text.split(" ") if word not in STOPWORDS)
  return text_processed


def remove_punctuation(text: str) -> str:
  remove_chars_list = ["?", ",", ".", ";", ":",  "!",'"']

  for char in remove_chars_list:
    text = text.replace(char, " ")

  return text


def lemmatisation(text: str) -> str:
  return " ".join(LEMMATIZER.lemmatize(word) for word in text.split(" "))


def remove_emoji(text: str) -> str:
  return emoji.replace_emoji(text, '')


def remove_short_words(text: str) -> str:
  return " ".join(word for word in text.split(" ") if len(word) > 2)


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
  df.loc[df["sentiment"] == "good", "target"] = 1
  df.loc[df["sentiment"] == "bad", "target"] = 0

  return df

def process_text(df: pd.DataFrame) -> pd.DataFrame:
  # [0] Combinando titulo y la descripción
  df["feature"] = df["reviewTitle"]  + " " + df["reviewDescription"]
  # [1] Eliminando signos de puntuación
  df["feature"] = df['feature'].apply(remove_punctuation)
  # [2] Convirtiendo a minúscula
  df["feature"] = df['feature'].str.lower()
  # [3] Eliminando las palabras más comunes
  df["feature"] = df['feature'].apply(remove_stopwords)
  # [4] Lematización
  df["feature"] = df['feature'].apply(lemmatisation)
  # [5] Removiendo emojis
  df["feature"] = df['feature'].apply(remove_emoji)
  # [6] Filtrando palabras muy cortas
  df["feature"] = df['feature'].apply(remove_short_words)
  # [7] Eliminando espacios en blanco al inicio y final del texto
  df["feature"] = df['feature'].str.strip()

  return df

# Aplicando el procesamiento a datos de entrenamiento
train_processed = process_text(train)
train_processed = encode_categories(train_processed)

# Revisión de la limpieza del texto
train_processed[["reviewTitle", "reviewDescription", "feature", "target"]].sample(5)

"""# Information Processing (vectorization)

A continuación procedere a vectorizar el texto para que pueda alimentar al modelo.
"""

# Term Frequency Inverse Document Frequency
# Considera el peso de las palabras y es una opción
# popular para la vectorización del text
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(train_processed["feature"])

df_x_train = pd.DataFrame.sparse.from_spmatrix(train_vec)
df_x_train.sample(5)

# Guardar el vectorizar en un archivo pickle
import pickle

VECTORIZER_PATH = '/content/vectorizer'
file = open(VECTORIZER_PATH, 'wb')
pickle.dump(vectorizer, file)
file.close()

from typing import Any

def save_to_pickle(obj: Any, path: str) -> None:
  file = open(path, 'wb')
  pickle.dump(obj, file)
  file.close()

def vectorize_features(df: pd.DataFrame, ) -> pd.DataFrame:
  file = open(VECTORIZER_PATH, 'rb')
  vectorizer = pickle.load(file)
  file.close()

  train_vec = vectorizer.transform(df)
  df_vectorized = pd.DataFrame.sparse.from_spmatrix(train_vec)

  return df_vectorized

# Resumen del vocabulario aprendido
# data_vec = {k: [v] for k, v in vectorizer.vocabulary_.items()}
# pd.DataFrame.from_dict(data_vec, orient="index").sort_values(0, ascending=False).head(10)

# [1] Preprocseamiento (limpieza del texto)
test_processed = process_text(test)
# [2] Vectorización
df_x_test = vectorize_features(test_processed["feature"])

df_x_test.head()

"""# Model training

## Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Obteniendo y_train para el entrenamiento del modelo
y_train = train_processed["target"]

parameters = {
    'penalty' : ['l1','l2'],
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}

logreg = LogisticRegression()

# GridSearchCV
# Crea un modelo por cada combinación del conjunto de parametros
# definidos, y los evalúa usando cross validation.
# Genera y evalua un total de 210 modelos y al final elegimos el mejor

logreg_gs = GridSearchCV(
    estimator=logreg,
    param_grid = parameters,
    # Una buen métrica para evaluar el modelo es f1-score,
    # ya que los datos se encuentran desbalaceados
    scoring='f1_macro',
    cv=5,
    n_jobs=-3,
    verbose=10,
)

logreg_gs.fit(df_x_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Aplicando a los datos de test
y_pred_train = logreg_gs.best_estimator_.predict(df_x_train)

# En la matriz de confusión vemos que
# logra predecir correctamente los 16 comentarios negativos
confusion_matrix(y_pred_train, y_train)

print(classification_report(y_pred_train, y_train))

y_pred_train = logreg_gs.best_estimator_.predict(df_x_train)

train_score = f1_score(y_train, y_pred_train, average="macro")
print("Tuned Hyperparameters :", logreg_gs.best_params_)
# Los valores de f1-score van de 0 a 1, mientras
# más cercano sea a 1, mejor será el modelo
print("F1 score en Train:", round(train_score, 2))

"""Tiene un alto score, por lo que es posible que se encuentre sobre entrenado.

### Validation
"""

# [1] Preprocesamiento (limpieza del texto)
test_processed = process_text(test)
# [2] Vectorización
x_test = test_processed["feature"]
df_x_test = vectorize_features(x_test)
# Aplicando a los datos de test
y_pred_test = logreg_gs.best_estimator_.predict(df_x_test)

test_processed = encode_categories(test)
y_test = test_processed["target"]

# En la matriz de confusión vemos que solo
# logra predecir 1 de los 4 comentarios negativos.
confusion_matrix(y_pred_test, y_test)

print(classification_report(y_pred_test, y_test))

test_score = f1_score(y_test, y_pred_test, average="macro")

print(f"Train score: {round(train_score, 2)} \nTest score: {round(test_score, 2)}")

"""El score de entrenamiento es muy alto, mientras que en los datos de test es de 0.7, es decir, más bajo, por lo que podemos determinar que hay overfitting.

Este problema podrá tratarse en otra iteración, con el uso de más datos,aumentado el número de k-folds, o utilizando un ensamblado de modelos.

Por ahora, el modelo tiene un desempeño acetable, ya que tiene un **score en los datos de Test de 69%**.
"""

# Guardando el modelo
LR_ESTIMATOR_PATH = "/content/lr_estimator"
save_to_pickle(logreg_gs.best_estimator_, LR_ESTIMATOR_PATH)

"""## XGBoost"""

# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range(2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

# En total se prueban 480 modelos
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=10,
)

grid_search.fit(df_x_train, y_train)

#grid_search.best_estimator_

y_pred_train = grid_search.best_estimator_.predict(df_x_train)

train_score = f1_score(y_train, y_pred_train, average="macro")

print("Tuned Hyperparameters :", grid_search.best_params_)
print(f"F1-score Train: {train_score}")

"""### Validación"""

y_pred = grid_search.best_estimator_.predict(df_x_test)

confusion_matrix(y_pred, y_test)

print(classification_report(y_pred, y_test))

test_score = f1_score(y_test, y_pred, average="macro")

print(f"Train score: {train_score} \nTest score: {test_score}")

"""Al revisar los scores, es posible decir que también se dio un overfitting, o sobreentreamiento de los modelos. Se ajusto muy bien en los datos de train, pero no tuvo un buen desempeño en los datos de test."""

# Guardando el modelo
XGB_ESTIMATOR_PATH = "/content/xgb_estimator"
save_to_pickle(logreg_gs.best_estimator_, LR_ESTIMATOR_PATH)

"""Comparando el score para ambos modelos (Logistic Regression y XGB), el que tuvo mejor desempeño fue **Logistic Regession**, debido a tuvo un mejor score en los datos de test.

# Function to process and predict new data
"""

#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""Programa que realiza determinar si una reseña es positiv o negativa para"""

__author__ = "Cuauhtémoc"
__version__ = "1.0"
__status__ = "Development"

import pandas as pd

import emoji
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Descargando librería stopwords
nltk.download('stopwords')
# Descargando librería para lematización
nltk.download('wordnet')

# Obteniendo stopwords para idioma inglés
STOPWORDS = nltk.corpus.stopwords.words('english')
VECTORIZER_PATH = '/content/vectorizer'
LR_ESTIMATOR_PATH = "/content/lr_estimator"
XGB_ESTIMATOR_PATH = "/content/xgb_estimator"
BEST_MODEL_PATH = LR_ESTIMATOR_PATH


LEMMATIZER = WordNetLemmatizer()

def remove_stopwords(text: str) -> str:
  text_processed = " ".join(word for word in text.split(" ") if word not in STOPWORDS)
  return text_processed


def remove_punctuation(text: str) -> str:
  remove_chars_list = ["?", ",", ".", ";", ":",  "!",'"']

  for char in remove_chars_list:
    text = text.replace(char, " ")

  return text


def lemmatisation(text: str) -> str:
  return " ".join(LEMMATIZER.lemmatize(word) for word in text.split(" "))


def remove_emoji(text: str) -> str:
  return emoji.replace_emoji(text, '')


def remove_short_words(text: str) -> str:
  return " ".join(word for word in text.split(" ") if len(word) > 2)


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
  df.loc[df["sentiment"] == "good", "target"] = 1
  df.loc[df["sentiment"] == "bad", "target"] = 0

  return df


def process_text(df: pd.DataFrame) -> pd.DataFrame:
  # [0] Combinando titulo y la descripción
  df["feature"] = df["reviewTitle"]  + " " + df["reviewDescription"]
  # [1] Eliminando signos de puntuación
  df["feature"] = df['feature'].apply(remove_punctuation)
  # [2] Convirtiendo a minúscula
  df["feature"] = df['feature'].str.lower()
  # [3] Eliminando las palabras más comunes
  df["feature"] = df['feature'].apply(remove_stopwords)
  # [4] Lematización
  df["feature"] = df['feature'].apply(lemmatisation)
  # [5] Removiendo emojis
  df["feature"] = df['feature'].apply(remove_emoji)
  # [6] Filtrando palabras muy cortas
  df["feature"] = df['feature'].apply(remove_short_words)
  # [7] Eliminando espacios en blanco al inicio y final del texto
  df["feature"] = df['feature'].str.strip()

  return df


def vectorize_features(df: pd.DataFrame, ) -> pd.DataFrame:
  file = open(VECTORIZER_PATH, 'rb')
  vectorizer = pickle.load(file)
  file.close()

  train_vec = vectorizer.transform(df)
  df_vectorized = pd.DataFrame.sparse.from_spmatrix(train_vec)

  return df_vectorized

# Flujo para cargar el modelo y realizar predicción
import sklearn
def load_model():
  file = open(BEST_MODEL_PATH, 'rb')
  estimator = pickle.load(file)
  file.close()

  return estimator


def main(df):

  # [1] Preprocseamiento (limpieza del texto)
  df_processed = process_text(df)
  # [2] Vectorización
  features = vectorize_features(df_processed["feature"])
  # [3] Cargando el modelo
  estimator = load_model()
  # [4] Realizando la predicción
  y = estimator.predict(features)[0]

  sentiment = "good"
  if y == 0:
    sentiment = "bad"

  return sentiment

"""# Simulación de la llegada de nuevos datos"""

df_raw = pd.read_csv("/content/flipkart_review_data_2022_02.csv")
df_raw = df_raw.drop(columns="Unnamed: 0")

df_raw["sentiment"] = df_raw["reviewTitle"].apply(relabel_target)

df_test = pd.concat([df_raw[df_raw["sentiment"] == "bad"].sample(5),
                    df_raw[df_raw["sentiment"] == "good"].sample(5)])

df_test.groupby("sentiment")["sentiment"].count()

import time

for index, row in df_test.iterrows():
  sentiment = main(pd.DataFrame(row).T)

  print(f"Label: {row['reviewTitle']}", f"-> Prediction: {sentiment}")

  time.sleep(1)

"""# Explanation of approach and decisions

El primer paso fue realizar fue realizar un filtrado de los comentarios, para obtener los comentarios de mejor calidad. Esto al eliminar el 1% de los comentarios con más dislikes.

El siguiente paso fue categorizar y crear una etiqueta para nuestros datos y puedan alimentar a los modelos, para ello utilicé como base, los titulos de los reviews.

Generé dos etiquetas: "good" y "bad". Al revisar la distribución de los datos, noté que la etiqueta "good" caracteriza a la mayoría de comentarios, por lo tanto los datos se encuentran desbalanceados.

El siguiente paso fue la obtención de los datos de train y test, normalmente se usa un 70% para los datos de entrenamiento y un 30% para los datos de test. Pero al contar con muy pocos registros, decidí usar 80-20, para tener una mayor cantidad de dastos para el entrenamiento.

Para tratar datos no balanceados, se puede:
- Realizar un sub-muestreo de la data que tiene mayor cantidad de datos
- Realizar un sobre-muestreo de la etiqueta con menor cantidad de datos
- Usar stratify cross validation
- Ensamblado de diferentes modelos, realizando un submuestreo de la categoría con mayor cantidad de datos y siempre usando la categoría con menos datos

Por el tiempo disponible opté por usar stratify cross validation que se integra dentro de la función GridSearch.

Tanto la columna tilte como reviewDescription fueron concatenadas, para ser utilizadas como "feature" para la construcción del modelo, el preprocesamiento incluyo:

- Eliminar signos de puntuación
- Convertir a minúsculas
- Remover palabras más comunes (stopwords)
- Eliminación de emojis
- Lematización: llevar a la raiz
- Remover palabras menores a 2 letras
- Vectorizar la información

Ya que es un problema de clasificación, use dos modelos de clasificación supervisada, Logistic Regression y XGB. Use diferentes hiperparametros que generaron diversos submodelos y poder evaluar una mayor cantidad de estos, mediante f1 score.

Una buen métrica para evaluar el modelo es f1-score, ya que los datos se encuentran desbalaceados

En ambos modelos tuvieron un buen desempeño con los datos de entrenamiento, pero no muy bien con los datos de train, por lo que se dio un overfitting (sobreentramiento).

El modelo Logistic Regression tiene un desempeño acetable.

En cuanto a oportunidades de mejora están:
- Utilizar los emojis dentro del modelo
- Tener una mayor cantidad de datos
- Evaluar la obtención de las etiquetas "good" y "bad"
- Probar técnias de reducción de dimensionalidad (PCA)
- Probar con otros parámetros al vectorizar con la función TfidfVectorizer
- Evaluar más algoritmos e hiperparametros: svm, knn, random forest, naives bayes
- Mejorar el score de manera general, y tener cuidado con el overfitting.

"""