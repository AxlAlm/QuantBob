
#basics
import numpy as np


# from:
# https://docs.numer.ai/tournament/learn#scoring
# and
# https://github.com/numerai/example-scripts

def era_correlation(y_true, y_pred):
    rank_pred = y_pred.groupby("eras").apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]


def correlation_score(y_true, y_pred):
    ranked_pred = y_pred.rank(pct=True, method="first")
    return np.corrcoef(y_true, ranked_pred)[0,1]


