# Data Preprocessor
from scipy.io import arff
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder, MinMaxScaler


def load_data_from_arff(path):
   """
  loads the given data from arff file located in the given path. returns pandas DataFrame
  """
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    return df


def plot_discriminetor_generator_hist_graph(history):
    plt.plot(history['g_loss'], label="generator")
    plt.plot(history['d_loss'], label="disciminator")
    plt.title('log')
    plt.grid(True)
    plt.legend()
    plt.show()


def preprocess_data(data_df, normalized=True, how='standard', class_col='class', resample=True):
    """
  transform and normalize the data according to the how argument. gets a data_df and returns a transformed pandas dataframe.
  """
    les = {}
    data_columns = data_df.columns[data_df.columns != class_col].tolist()
    transformed_columns = []
    for col in data_columns:
        if data_df[col].dtype != np.float64:
            transformed_columns.append(col)
            les[col] = LabelEncoder()
            les[col].fit(data_df[col])
            data_df[col] = les[col].transform(data_df[col])
    normalizer = None
    if normalized:
        if how == 'standard':
            normalizer = StandardScaler().fit(data_df[data_columns])
        elif how == 'power':
            normalizer = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(data_df[data_columns])
        elif how == "max_min_scalar":
            normalizer = MinMaxScaler().fit(data_df[data_columns])
        data_df[data_columns] = normalizer.transform(data_df[data_columns])
        if resample:
            oversample = SMOTE()
            X, y = oversample.fit_resample(data_df.iloc[:, :-1], data_df.iloc[:, -1])
            data_df_tmp = pd.DataFrame(data=X)
            data_df_tmp['y'] = y
            data_df_tmp.columns = data_df.columns
            data_df = data_df_tmp.copy()
    return data_df, les, normalizer, transformed_columns


def inverse_transform(data_df, les, normalizer, transformed_columns, normalized=True, class_col='class'):
    data_columns = data_df.columns[data_df.columns != class_col].tolist()
    if normalized:
        data_df[data_columns] = normalizer.inverse_transform(data_df[data_columns])
    for col in transformed_columns:
        data_df[col] = les[col].inverse_transform(data_df[col].astype(int))
    return data_df
