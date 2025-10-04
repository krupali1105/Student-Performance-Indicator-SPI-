import os
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.config.model_trainer_config import ModelTrainerConfig
from src.utils import save_obj, evaluate_models
from src.config.hyper_parameter_tuning_config import HyperParameterTuningConfig

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()

    def init_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting Training and Input Test data")
            X_train, y_train, X_test, y_test =(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = HyperParameterTuningConfig().params
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found!!")
            
            logging.info(f"Best model found on both Training and Testing Dataset: {best_model_name}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            cal_r2_score = r2_score(y_test, predicted)

            return cal_r2_score

        except Exception as e:
            raise CustomException(e, sys)