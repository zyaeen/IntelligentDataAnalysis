import numpy as np

from DataPreparing import prepare_data
from RMSE import rmse


def naive_predict(top, data):
    # Структура для хранения для каждого пользователя оценки фильмов top наиболее похожих на него пользователей:
    # top_similar_ratings[0][1] - оценки всех фильмов одного из наиболее похожих пользователей на пользователя с ид 0.
    # Здесь 1 - это не ид пользователя, а просто порядковый номер.
    top_similar_ratings = np.zeros((data["n_users"], top, data["n_movies"]))

    for i in range(data["n_users"]):
        # Для каждого пользователя необходимо получить наиболее похожих пользователей:
        # Нулевой элемент не подходит, т.к. на этом месте находится похожесть пользователя самого на себя
        top_sim_users = data["user_similarity"][i].argsort()[1:top + 1]

        # берём только оценки из "обучающей" выборки
        top_similar_ratings[i] = data["train_data_matrix"][top_sim_users]

    pred = np.zeros((data["n_users"], data["n_movies"]))
    for i in range(data["n_users"]):
        pred[i] = top_similar_ratings[i].sum(axis=0) / top

    return pred


def naive_predict_item(top, data):
    top_similar_ratings = np.zeros((data["n_movies"], top, data["n_users"]))

    for i in range(data["n_movies"]):
        top_sim_movies = data["item_similarity"][i].argsort()[1:top + 1]
        top_similar_ratings[i] = data["train_data_matrix"].T[top_sim_movies]

    pred = np.zeros((data["n_movies"], data["n_users"]))
    for i in range(data["n_movies"]):
        pred[i] = top_similar_ratings[i].sum(axis=0) / top

    return pred.T


prepared_data = prepare_data(path='S:/Проекты 5 курс/Python/Датасеты/ratings.csv',
                             sample_size=100_000,
                             test_size=0.2
                             )

naive_pred = naive_predict(7, prepared_data)
print('User-based CF RMSE: ', rmse(naive_pred, prepared_data["test_data_matrix"]))

naive_pred_item = naive_predict_item(7, prepared_data)
print('Item-based CF RMSE: ', rmse(naive_pred_item, prepared_data["test_data_matrix"]))