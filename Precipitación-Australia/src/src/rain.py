import pickle
import pandas as pd
import collections
from typing import List, Dict, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from sklearn import metrics
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Creando un tipo de objeto para los clasificadores
ModelRegressor = Union[SVC, KNeighborsClassifier, DecisionTreeClassifier,
                       GaussianNB, MultinomialNB, ComplementNB,
                       LogisticRegression, XGBClassifier]


def read_weather(path: str) -> pd.DataFrame:
    """Reads the csv file with the weather data from Australia,
    and the column "Date" is processed"""
    df = pd.read = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month

    return df


def missing_data(data: pd.DataFrame) -> pd.Series:
    """Returns a Series with the percent missing data  by column
    in a dataframe"""
    return round(data.isnull().sum() / len(data)
                 * 100, 2)


def missing_data_by_year(data: pd.DataFrame,
                         column: str) -> pd.Series:
    """Return the missing data percente group by year for the
    given column"""
    return round(data[column].isnull()
                 .groupby(data['year']).sum()
                 / data['year'].groupby(data['year']).count()
                 * 100, 2)


def encode_sin_cos(data: pd.DataFrame, col, max_val) -> pd.DataFrame:
    """Transform the values using sinus and cosine to convert the values
    to a cyclical values"""
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data


def pipline_read_data(path: str) -> pd.DataFrame:
    """Returns the Australia Weather dataframme with some preprocess:
    dropping rows with missing data in the target attribute 'RainTomorrow',
    and encoding the month with the sinus and cosine transformations"""
    df = read_weather(path)
    df = df.dropna(subset=['RainTomorrow'])
    df = encode_sin_cos(df, 'month', 12)

    return df


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data in train and test, with a proportion of 0.75 (train)
     and 0.25 (test) in a stratify form """

    df_train, df_test = train_test_split(data,
                                         train_size=0.75, random_state=1,
                                         stratify=data['RainTomorrow'])

    return df_train, df_test


def save_to_pickle(obj: object, path: str) -> None:
    """"Save an object in a pickle file"""
    with open(path + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> any:
    """Reads a pickle file"""
    with open(path, 'rb') as file:
        object_file = pickle.load(file)
    return object_file


def get_mode(series: pd.Series) -> str:
    """Return the mode for a dataset"""
    return series.mode()[0]


def get_median(series: pd.Series) -> float:
    """Return the median for a dataset"""
    return round(series.median(), 2)


def create_dictionary_mode(data: pd.DataFrame,
                           columns: list[str, ...]) -> Dict:
    """Returns a dictionarty with the mode for every column in a dataframe"""
    return {col: get_mode(data[col]) for col in columns}


def create_dictionary_median(data: pd.DataFrame,
                             columns: list[str, ...]) -> Dict:
    """Returns a dictionarty with the median for every column in a dataframe"""
    return {col: get_median(data[col]) for col in columns}


def get_categorical_columns(data: pd.DataFrame) -> list:
    """Returns a list with the names of the columns that are object types"""
    return data.select_dtypes('object').columns.to_list()


def get_numerical_columns(data: pd.DataFrame) -> list:
    """Returns a list with the names of the columns that are float types"""
    return data.select_dtypes('float64').columns.to_list()


def encode_target(series: pd.Series) -> pd.Series:
    """Map the target attribute from 'Yes', 'No' to 1, 0"""
    return series.map(dict(Yes=1, No=0))


def encode_dummies(data: pd.DataFrame) -> pd.DataFrame:
    """Encode the categorical variables into dummies"""

    return pd.get_dummies(data, drop_first=True)


def create_dictionary_outliers(data: pd.DataFrame,
                               threshold: float = 0.95) -> dict:
    """Returns a dictionarty with the lower and upper value get from IQR 0.5
    and 0.95 for every column in a dataframe"""

    outliers_dict = {}

    for col in get_numerical_columns(data):
        lower = data[col].quantile(1 - threshold)
        upper = data[col].quantile(threshold)
        outliers_dict[col] = {'lower': lower,
                              'upper': upper}

    return outliers_dict


def processing_outliers(data: pd.DataFrame,
                        outliers_dict: dict) -> pd.DataFrame:
    """Converts the values under the lower value into the lower value, and
        the values above the upper value into the upper value.

        The lower and upper values were obtained through IQR 0.5 and 0.95"""

    for key in outliers_dict:
        if key in data.columns:
            data.loc[data[key] < outliers_dict[key]['lower'], key] = \
                outliers_dict[key]['lower']
            data.loc[data[key] > outliers_dict[key]['upper'], key] = \
                outliers_dict[key]['upper']

    return data


def process_data(data: pd.DataFrame,
                 drop_columns: Union[list, None] = None) -> pd.DataFrame:
    """flow to process the data set to fine-tune for modeling.
    The process consists:
    - Filling with the mode for categorical missing data
    - Filling with the median for numerical missing data
    - Encoding the target
    - Dropping useless columns and attributes with multicollineartiy
    - Dealing with outliers"""

    # They're eliminated for missing data (>35%)
    # data = data.drop(
    # columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'])

    # Filling nan values with mode for categorical values
    mode_dict = load_pickle('../results/mode_dict.pickle')
    median_dict = load_pickle('../results/median_dict.pickle')

    # Filling nan values with mode for categorical values
    for col in mode_dict.keys():
        if col in data.columns:
            data[col].fillna(mode_dict[col], inplace=True)
    # Filling nan values with median for numerical values
    for col in median_dict.keys():
        if col in data.columns:
            data[col].fillna(median_dict[col], inplace=True)

    # Encoding attribute target
    data['RainTomorrow'] = encode_target(data['RainTomorrow'])

    if drop_columns:
        data = data.drop(columns=drop_columns)

    # Processing outliers
    outliers_dict = load_pickle(r'..\results\outliers_dict.pickle')
    data = processing_outliers(data, outliers_dict)

    return data


def get_santandar_scaler(data: pd.DataFrame) -> StandardScaler:
    """Return a StandarScaler object fitted for a given dataset"""

    scalar = StandardScaler()
    scalar.fit(data)

    return scalar


def plot_cm(model, x: pd.DataFrame, y: pd.Series) -> None:
    """Plot a normalized confusion matrix"""
    plot_confusion_matrix(model, x, y, display_labels=['Not Rain', 'Rain'],
                          cmap=plt.cm.Blues, normalize='true')


def filter_categorical_variables(data: pd.DataFrame,
                                 threshold: Union[
                                     int, None] = None) -> pd.DataFrame:
    """Remove the columns with high cardinality, this is the
    unique values in the column es greater than threshold"""

    if threshold:
        categorical_cols = get_categorical_columns(data)
        for col in categorical_cols:
            if len(data[col].unique()) > threshold:
                data.drop(columns=[col], inplace=True)

    return data


def pipline_process_data(drop_columns: Union[list, None] = None):
    """"""
    file = r'..\data\weatherAUS.csv'
    df = pipline_read_data(file)  # Leyendo el dataset
    df_train, df_test = split_data(df)  # Particionando la información
    # Procesando los conjuntos de dato de datos
    df_train = process_data(df_train, drop_columns=drop_columns)
    # Procesando los conjuntos de dato de datos
    df_test = process_data(df_test, drop_columns=drop_columns)

    # Particionando los datos en X y Y
    y_train = df_train.pop('RainTomorrow')
    X_train = df_train
    y_test = df_test.pop('RainTomorrow')
    X_test = df_test

    # Codificando en variables dummies
    X_train = encode_dummies(X_train)
    X_test = encode_dummies(X_test)
    # Transformando los valores númericos
    numerical_cols = get_numerical_columns(X_train)
    scalar = load_pickle(r'../results/standarscaler.pickle')
    # Aplicando la transformación a los conjuntos de datos de X de
    # entrenamiento y prueba
    X_train[numerical_cols] = scalar.transform(X_train[numerical_cols])
    X_test[numerical_cols] = scalar.transform(X_test[numerical_cols])

    return X_train, y_train, X_test, y_test


def get_features_by_xgb_importance(
        model: XGBClassifier, importance_type: str) -> List:
    """Returns a list with the features sorted by importance"""

    imp_scores_d = model.get_booster().get_score(
        importance_type=importance_type)

    sorted_imp = sorted(imp_scores_d.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_imp)

    return [key for key in sorted_dict.keys()]


def estimate_score_metrics(y_test: pd.Series,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray
                           ) -> Tuple[float, float, int, int, int, int]:
    """Returns the following evaluation metrics: ROC, ROC_AUC,
    \rF1-score, Recall, Accuracy, Brier"""
    roc = round(metrics.roc_auc_score(y_test, y_pred), 2)
    roc_auc = round(metrics.roc_auc_score(y_test, y_prob), 2)

    f1 = round(metrics.f1_score(y_test, y_pred) * 100)
    recall = round(metrics.recall_score(y_test, y_pred) * 100)
    accuracy = round(metrics.accuracy_score(y_test, y_pred) * 100)
    brier = round(metrics.brier_score_loss(y_test, y_pred) * 100)

    return roc, roc_auc, f1, recall, accuracy, brier


def get_total_iterations(model, importance_types_list: List) -> int:
    """Returns the total of iterations for the modeling by xgb
    feature importance"""
    no_elements = 1
    for imp_type in importance_types_list:
        features = get_features_by_xgb_importance(
            model=model, importance_type=imp_type)

        while len(features) > 0:
            _ = features.pop(0)
            no_elements += 1

    return no_elements


def modeling_by_subset(
        model: ModelRegressor,
        x_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        features: List) -> np.array:
    """Returns an array with Machine Learning model, identifier of model,
    number of features, metrics scores, """
    return model.fit(x_train[features], y_train)


def predict_by_subset(
        predictor: ModelRegressor,
        x_test: pd.DataFrame,
        features: List) -> np.array:
    """Returns an array with Machine Learning model, identifier of model,
    number of features, metrics scores, """
    x_test_subset = x_test[features]
    y_pred = predictor.predict(x_test_subset)
    y_prob = np.around(predictor.predict_proba(x_test_subset)[:, 1], 2)

    return y_pred, y_prob


def modeling_by_xgb_importance(
        model_name: str,
        model: ModelRegressor,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test) -> pd.DataFrame:
    """Returns a dataframe of models scores using xgboost feature
    \r importance to select the best features
    """
    gral_model = XGBClassifier(n_jobs=-1)
    gral_model_fitted = gral_model.fit(x_train, y_train)
    imp_types_lst = ['total_gain', 'total_cover', 'weight', 'gain', 'cover']

    no_elements = get_total_iterations(gral_model_fitted, imp_types_lst)
    count = 1
    row_lst: List[np.array] = []
    row_array = np.array(row_lst)

    for importance_type in imp_types_lst:
        features_list = get_features_by_xgb_importance(
            model=gral_model_fitted, importance_type=importance_type)

        while len(features_list) > 0:

            predictor = modeling_by_subset(model=clone(model),
                                           x_train=x_train,
                                           y_train=y_train,
                                           features=features_list)

            y_pred, y_prob = predict_by_subset(predictor=predictor,
                                               x_test=x_test,
                                               features=features_list)

            score_metrics = estimate_score_metrics(
                y_test=y_test, y_pred=y_pred, y_prob=y_prob)

            row_lst.append(np.array([
                model_name,
                'model_' + str(count),
                len(features_list),
                *score_metrics,
                importance_type,
                ','.join(features_list)]))

            count += 1
            _ = features_list.pop(0)

            txt = 'of ' + ' Modeling with ' + str(model_name) + ' :'
            update_progress(count / no_elements, progress_text=txt)

        row_array = np.array(row_lst)

    # Names of columns of info value dataframe
    cols_dict = {
        'Models': str, 'Id': str, 'No_features': int, 'ROC': float,
        'ROC_AUC': float, 'F1': float, 'Recall': float, 'Accuracy': float,
        'Brier': float, 'Importance_type': str, 'Best_features': str}

    cols_name = [names for names in cols_dict]
    df = pd.DataFrame(data=row_array, columns=cols_name)
    df = df.astype(cols_dict)

    df = df.sort_values(['F1', 'ROC_AUC', 'No_features'],
                        ascending=False).reset_index(drop=True)

    clear_output(wait=False)

    return df


def update_progress(progress, progress_text=''):
    """ Print the progress of a 'FOR' inside a function """

    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    progress = max(progress, 0)
    progress = min(progress, 1)
    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = ' '.join(['Progress', progress_text, '[{0}] {1:.1f}%'])
    ouput_text = text.format("#" * block + "-" * (bar_length - block),
                             progress * 100)

    print(ouput_text)


