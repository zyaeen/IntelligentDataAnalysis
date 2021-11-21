import numpy as np

from DataPreparing import prepare_data
from RMSE import rmse


# TODO: Хабиб, тут берешь, значит, ту статью,
#  и по разделу "Рекомендации на основе средних оценок пользователей и матрицы “похожести”"
#  вот сюда пишешь код в две функции ниже.
#  Только помни, что переменные user_similarity, item_similarity,
#  n_users, n_movies, train_data_matrix и test_data_matrix нужно
#  вызывать как ключи словаря: data["user_similarity"], data["n_users"]...


def k_fract_mean_predict(top, data):

    pred = np.zeros((data["n_movies"], data["n_users"]))

    return pred


def k_fract_mean_predict_item(top, data):

    pred = np.zeros((data["n_movies"], data["n_users"]))

    return pred


prepared_data = prepare_data(path='S:/Проекты 5 курс/Python/Датасеты/ratings.csv',
                             sample_size=100_000,
                             test_size=0.2
                             )

pred = k_fract_mean_predict(7, prepared_data)
print('User-based CF RMSE: ', rmse(pred, prepared_data["test_data_matrix"]))

pred_item = k_fract_mean_predict_item(7, prepared_data)
print('Item-based CF RMSE: ', rmse(pred_item, prepared_data["test_data_matrix"]))