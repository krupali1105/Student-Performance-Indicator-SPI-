from dataclasses import dataclass, field

def default_hyper_parameter_tuning_config():
    return {
        "Random Forest": {
            'n_estimators': [10, 30, 100, 300, 1000, 1500],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_features': ['sqrt', 'log2', None]
        },
        "Decision Tree": {
            'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter':['best','random'],
            'max_features':['sqrt','log2'],
        },
        "Gradient Boosting": {
            'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            'criterion':['squared_error', 'friedman_mse'],
            'max_features':['sqrt','log2', None],
            'n_estimators': [10, 30, 100, 300, 1000, 1500]
        },
        "Linear Regression": {},
        "XGBRegressor": {
            'learning_rate':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'n_estimators': [10, 30, 100, 300, 1000, 1500]
        },
        "CatBoosting Regressor": {
            'depth': [6,8,10],
            'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'iterations': [30, 50, 100]
        },
        "AdaBoost Regressor": {
            'learning_rate':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'loss':['linear','square','exponential'],
            'n_estimators': [10, 30, 100, 300, 1000, 1500]
        },
    }

@dataclass
class HyperParameterTuningConfig:
    params: dict = field(default_factory=default_hyper_parameter_tuning_config)