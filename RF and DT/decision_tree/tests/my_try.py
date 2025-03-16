import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from decision_tree.decision_tree import ClassificationDecisionTree
from decision_tree.decision_tree import RegressionTree



np.random.seed(1)
reg_tree = RegressionTree(max_depth=4, min_leaf_size=1)
x_shape = 300
x = np.arange(x_shape) / 100
y = x ** 3 * np.sin(x ** 3) + np.random.random(x_shape)

reg_tree.fit(x.reshape(-1, 1), y)

        # Прогнозы
predictions = reg_tree.predict(x.reshape(-1, 1))
print(np.abs((predictions - y)).mean())
print(predictions)
        # Проверка, что средняя ошибка модели на тестовых данных мала
plt.plot(x, y, label='Истинные значения', color='blue', linestyle='-', linewidth=2)
plt.plot(x, predictions, label='Предсказания', color='red', linestyle='--', linewidth=2)

