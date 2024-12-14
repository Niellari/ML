from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', repl_value:float=np.nan, low_q=0.25, high_q=0.75, threshold=1.5):
        """
        Трансформер, реализующий замену вылетов на указанное значение.
        Вылеты определяются двумя методами: межквантильный размах (IQR) и на основе стд.отклонений (std)

        :param method: метод определения вылетов (outliers)
            - 'iqr': использовать межквантильный диапазон (IQR)
            - 'std': использовать стандартное отклонение
        :param repl_value: значение, на которое заменяются вылеты
        :param low_q: нижний квантиль для метода межквантильного диапазона
        :param high_q: верхний квантиль для метода межквантильного диапазона
        :param threshold: порог для определения вылетов (используется для всех методов)
        """
        self.method = method
        self.repl_value = repl_value
        self.low_q = low_q
        self.high_q = high_q
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X должен быть pandas.DataFrame.")

        if self.method == 'iqr':
            Q1 = X.quantile(self.low_q, numeric_only=True)
            Q3 = X.quantile(self.high_q, numeric_only=True)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - (IQR * self.threshold)
            self.upper_bound = Q3 + (IQR * self.threshold)
        elif self.method == 'std':
            mean = X.mean()
            std = X.std()
            self.lower_bound = mean - (self.threshold * std)
            self.upper_bound = mean + (self.threshold * std)

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X должен быть pandas.DataFrame.")

        # Создаем копию DataFrame, чтобы избежать изменения исходных данных
        transformed_X = X.copy()

        for col in transformed_X.columns:
            if self.method == 'iqr':
                transformed_X.loc[
                    (transformed_X[col] < self.lower_bound[col]) | (transformed_X[col] > self.upper_bound[col])
                    , col
                ] = self.repl_value
            elif self.method == 'std':
                transformed_X.loc[
                    (transformed_X[col] < self.lower_bound[col]) | (transformed_X[col] > self.upper_bound[col])
                    , col
                ] = self.repl_value

        return transformed_X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Вычисление статистик и трансформация данных.

        :param X: Массив признаков.
        :param y: Целевая переменная (не используется).
        :return: Новый массив признаков с замененными выбросами.
        """
        self.fit(X, y)
        return self.transform(X)


class Replacer(BaseEstimator, TransformerMixin):
    """
    Заменяет XNA на указанное значение.
    В целом, это класс, который может заменять значения по порогу.
    В основном использовался перед трансформацией Бокса-Кокса для замены нулевых значений
    Параметры
    ----------
    repl: float, по умолчанию 0.1
    Значение для замены.
    """

    def __init__(self, repl_value=0.1):
        self.repl_value = repl_value

    def fit(self, X, y=None):
        return self

    # transform выполняет всю работу: применяет преобразование
    # с помощью заданного значения параметра repl_value
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X[X == 'XNA'] = self.repl_value
        else:
            X = np.where(X == 'XNA', self.repl_value, X)
        return X


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Кастомный LabelEncoder для использования в Pipeline.
    Является обёрткой над scikit-learn.LabelEncoder чтоб можно было
    применять к pandas.DataFrame
    """
    def __init__(self):
        self.encoders = {}  # Словарь для хранения LabelEncoder для каждой колонки

    def fit(self, X, y=None):
        # Преобразуем X в DataFrame, если это массив numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Обучаем отдельный LabelEncoder для каждой колонки
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        # Преобразуем X в DataFrame, если это массив numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Преобразуем каждую колонку независимо
        X_transformed = X.copy()
        for col in self.encoders:
            X_transformed[col] = self.encoders[col].transform(X[col].astype(str))
        return X_transformed.values  # Возвращаем numpy массив для совместимости с другими шагами Pipeline

    def inverse_transform(self, X):
        # Преобразуем X в DataFrame, если это массив numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.encoders.keys())

        # Обратное преобразование для каждой колонки
        X_inverse = X.copy()
        for col in self.encoders:
            X_inverse[col] = self.encoders[col].inverse_transform(X[col].astype(int))
        return X_inverse.values  # Возвращаем numpy массив для совместимости


class RareCategoryCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, n_unique_threshhold=4, freq_threshold=0.01, rare_label="_other_"):
        """
        Кастомный трансформер для объединения редких категорий.

        :param n_unique_threshhold: Порог количества категорий. Если количество категорий меньше, то признак не рассматривается.
        :param freq_threshold: Порог минимальной доли категорий. Если доля категории меньше, она будет заменена на rare_label.
        :param rare_label: Метка для объединенных редких категорий.
        """
        self.n_unique_threshhold = n_unique_threshhold
        self.freq_threshold = freq_threshold
        self.rare_label = rare_label
        self.mapping_ = {}

    def fit(self, X, y=None):
        """
        Запоминает частотное распределение категорий в каждом столбце.

        :param X: Входные данные (DataFrame).
        :param y: Не используется.
        :return: self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Входные данные должны быть pandas DataFrame.")

        self.mapping_ = {}
        for col in X.select_dtypes(include=["object", "category"]).columns:
            if X[col].nunique() >= self.n_unique_threshhold:
                freq = X[col].value_counts(normalize=True)
                self.mapping_[col] = freq[freq >= self.freq_threshold].index.tolist()

        return self

    def transform(self, X):
        """
        Преобразует данные, заменяя редкие категории на rare_label.

        :param X: Входные данные (DataFrame).
        :return: Преобразованный DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Входные данные должны быть pandas DataFrame.")

        X_transformed = X.copy()
        for col, valid_categories in self.mapping_.items():
            X_transformed[col] = np.where(
                X_transformed[col].isin(valid_categories),
                X_transformed[col],
                self.rare_label
            )

        return X_transformed
