
# sklearn
from sklearn.utils import all_estimators
from sklearn.base import BaseEstimator

# xgb
from xgboost import XGBRegressor


def get_regressor(name:str) -> BaseEstimator:
    
    if name == "XGBRegressor":
        return XGBRegressor
    else:
        return dict(all_estimators())[name]
