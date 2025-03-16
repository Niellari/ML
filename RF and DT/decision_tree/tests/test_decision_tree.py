import numpy as np
import unittest
import matplotlib.pyplot as plt
from decision_tree.decision_tree import ClassificationDecisionTree
from decision_tree.decision_tree import RegressionTree

class TestDecisionTree(unittest.TestCase):
    def test_small_decision_tree(self):
        np.random.seed(1)
        clas_tree = ClassificationDecisionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(10, 2)),
            np.random.normal(loc=(-5, 5), size=(10, 2)),
            np.random.normal(loc=(5, -5), size=(10, 2)),
            np.random.normal(loc=(5, 5), size=(10, 2)),
        ))
        y = np.array(
            [0] * 20 + [1] * 20
        )
        clas_tree.fit(x, y)
        predictions = clas_tree.predict(x)
        assert (predictions == y).mean() == 1

    def test_decision_tree(self):
        np.random.seed(1)
        clas_tree = ClassificationDecisionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(100, 2)),
            np.random.normal(loc=(-5, 5), size=(100, 2)),
            np.random.normal(loc=(5, -5), size=(100, 2)),
            np.random.normal(loc=(5, 5), size=(100, 2)),
        ))
        y = np.array(
            [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100
        )
        clas_tree.fit(x, y)
        predictions = clas_tree.predict(x)
        assert (predictions == y).mean() == 1

class TestRegressionTree(unittest.TestCase):
    def test_small_decision_tree(self):
        np.random.seed(1)
        reg_tree = RegressionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(10, 2)),
            np.random.normal(loc=(-5, 5), size=(10, 2)),
            np.random.normal(loc=(5, -5), size=(10, 2)),
            np.random.normal(loc=(5, 5), size=(10, 2)),
        ))
        y = np.array(
            [0.1] * 10 + [0.5] * 10 + [1.0] * 10 + [1.5] * 10
        )
        reg_tree.fit(x, y)
        predictions = reg_tree.predict(x)

        plt.plot(x, y, label='Истинные значения', color='blue', linestyle='-', linewidth=2)
        #plt.plot(x, predictions, label='Предсказания', color='red', linestyle='--', linewidth=2)
        plt.title('Истинные значения и предсказания')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        # Показать график
        plt.show()

        assert np.abs((predictions - y)).mean() < 0.1
    def test2(self):
        np.random.seed(1)
        reg_tree = RegressionTree(max_depth=5, min_leaf_size=2)
        x_shape = 300
        x = np.arange(x_shape) / 100
        y = x ** 3 * np.sin(x ** 3) + np.random.random(x_shape)

        reg_tree.fit(x.reshape(-1, 1), y)

        # Прогнозы
        predictions = reg_tree.predict(x.reshape(-1, 1))
        print(np.abs((predictions - y)).mean())
        # Проверка, что средняя ошибка модели на тестовых данных мала

        plt.plot(x, y, label='Истинные значения', color='blue', linestyle='-', linewidth=2)
        plt.plot(x, predictions, label='Предсказания', color='red', linestyle='--', linewidth=2)
        plt.title('Истинные значения и предсказания')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        # Показать график
        plt.show()
        assert np.abs((predictions - y)).mean() < 1