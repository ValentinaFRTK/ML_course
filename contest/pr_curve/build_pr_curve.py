import numpy as np


def build_precision_recall_curve(
    true_labels: np.ndarray, predicted_probas: np.ndarray
) -> np.ndarray:
    """
    Данная функция строит PR-кривую для задачи бинарной классификации.
    В случае, когда нет ни одного объекта положительного класса функция должна вызывать ValueError().

    Args:
        true_labels (np.ndarray): Массив истинных меток класса. Состоит из 0 и 1.
            1 считается меткой положительного класса.
        predicted_probas (np.ndarray): Массив предсказанных вероятностей принадлежности объекта
            к положительному классу.
    
    Returns:
        np.ndarray: Массив размерами (len(true_labels)+1, 2), где в каждой строчке стоит пара (recall, precision), первым элементом всегда идет (0, 1)
    """

    # Проверяем, есть ли вообще положительные объекты в выборке
    total_positives = np.sum(true_labels)
    if total_positives == 0:
        raise ValueError("В true_labels нет ни одного положительного класса(")

    combined = np.column_stack((predicted_probas, true_labels))
    sorted_combined = combined[np.argsort(combined[:, 0])[::-1]] # сортируем по убыванию вероятности
    
    tp = 0
    fp = 0
    
    # список для хранения точек (recall, precision)
    curve_points = []
    
    # (recall=0, precision=1)
    curve_points.append([0.0, 1.0])

    for _, true_label in sorted_combined:
        if true_label == 1:
            tp += 1
        else: # true_label == 0
            fp += 1
        
        # текущие recall и precision
        recall = tp / total_positives
        # проверка на деление на ноль для precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        
        curve_points.append([recall, precision])
    
    return np.array(curve_points)