from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
import optuna
from typing import Callable

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class ModelBlender(BaseEstimator):
    def __init__(self, models, coefficients):
        self.models = models
        self.coefficients = coefficients

    def fit(self, X, y):
        for model in self.models:
            model.fit(X,y)

    def predict(self, X):

        weighted_predictions = np.zeros_like(self.models[0].predict_proba(X), dtype=np.float64)
        
        for model, coeff in zip(self.models, self.coefficients):
            model_predictions = model.predict_proba(X).astype(np.float64)
            weighted_predictions += coeff * model_predictions

        return (weighted_predictions[:, 1] >= 0.5)

def tune_and_blend_models(
    data: pd.DataFrame, 
    target_column: str, 
    metric: Callable,
    random_state: int = 42, 
    n_trials: int = 5,
    cv: int = 5
) -> ModelBlender:
    """
    Тюнинг моделей с помощью Optuna и их блендинг.

    :param data: DataFrame с данными.
    :param target_column: Название целевой колонки.
    :param random_state: Начальное значение для генератора случайных чисел.
    :param n_trials: Количество итераций для Optuna.
    :return: Обученный объект ModelBlender.
    """
    # Разделение данных
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Инициализация моделей
    model_names = ["CatBoost","LightGBM","XGBoost"]
    
    # Тюнинг моделей с помощью Optuna
    def tune_model(model_name: str):
        
        models_dict = {
            'CatBoost': CatBoostClassifier,
            'LightGBM': LGBMClassifier,
            'XGBoost': XGBClassifier,
        }

        def objective(trial):
        
            model_params = {
                'max_depth': trial.suggest_int("max_depth", 4, 10),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1e-1),
                'n_estimators': trial.suggest_int("n_estimators", 100, 500),
                'random_state': random_state 
            }

            model = models_dict[model_name](**model_params)

            score = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(metric))[0]
            
            return score
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(lambda trial: objective(trial), n_trials=n_trials)
        
        print(f"Лучшие параметры для {model_name}: {study.best_params}")

        model = models_dict[model_name](**study.best_params)

        model.fit(X, y)
        return model

    # блендинг
    def blend_models(models):
        def objective(trial):

            coefficients = [
                trial.suggest_float(f'coeff_{i}', 0, 1) for i in range(len(models))
            ]
        
            total_coeff = sum(coefficients)
            coefficients = [coeff / total_coeff for coeff in coefficients]

            combined_model = ModelBlender(models, coefficients)
            
            score = cross_val_score(combined_model, X, y, cv=cv, scoring=make_scorer(metric))[0]
            
            return score
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    
        study.optimize(lambda trial: objective(trial), n_trials=n_trials)
        
        best_coefficients = [study.best_params[f'coeff_{i}'] for i in range(len(models))]

        total_coeff = sum(best_coefficients)
        best_coefficients = [coeff / total_coeff for coeff in best_coefficients]

        combination = ModelBlender(models, best_coefficients)

        return combination

    tuned_models = []
    for name in model_names:
        tuned_models.append(tune_model(name))

    blender = blend_models(models=tuned_models)
    blender.fit(X,y)
    
    return blender