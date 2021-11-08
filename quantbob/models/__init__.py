


# sklearn
from sklearn.utils import all_estimators

# xgb
from xgboost import XGBRegressor

# all supported regressors
regressors = dict(all_estimators())
regressors["XGBRegressor"] = XGBRegressor