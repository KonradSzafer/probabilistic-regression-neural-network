import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from utils import get_abs_path


DATA_PATH = get_abs_path(1) / 'data'


def load_csv(filename: str, index_col=0):
    return pd.read_csv(
        DATA_PATH / filename,
        index_col=index_col
    )


def scale(X, y):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    return X, y, X_scaler, y_scaler


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True
    )
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    return X_train, X_test, y_train, y_test


def diamonds_dataset():
    # source: https://www.kaggle.com/code/dasaris/regression-diamond-price-prediction
    df = load_csv('diamonds.csv', index_col=0)
    # categorical to numeric
    cats = ['color', 'cut', 'clarity']
    for i in cats:
        df[i] = pd.factorize(df[i])[0]
    # drop dimensionless rows
    df = df.drop(df[df['x'] == 0].index)
    df = df.drop(df[df['y'] == 0].index)
    df = df.drop(df[df['z'] == 0].index)
    assert sum(list(df.isnull().sum())) == 0
    # split data
    X = df.drop('price', axis=1).to_numpy()
    y = df['price'].to_numpy().reshape(-1, 1)
    # scale and fit scalers
    X, y, X_scaler, y_scaler = scale(X, y)
    X_train, X_test, y_train, y_test = split(X, y)
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_scaler': X_scaler,
        'y_train': y_train,
        'y_test': y_test,
        'y_scaler': y_scaler
    }


def load(name: str):
    if name == 'diamonds':
        return diamonds_dataset()
    else:
        raise ValueError(f'{name} dataset is not defined')
