

#basics 
import numpy as np
import pandas as pd


"""
from https://github.com/numerai/example-scripts/blob/0a8c4f764a3aee3b7c1709058dd1488b26bd5f01/analysis_and_tips.ipynb
"""


# The models should be scored based on the rank-correlation (spearman) with the target
def numerai_score(y_true, y_pred):
    rank_pred = y_pred.groupby("eras").apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

# It can also be convenient while working to evaluate based on the regular (pearson) correlation
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)
    

def validation_metrics(validation_data, fast_mode=False):
    """
    This is just a refactored version of :

    https://github.com/numerai/example-scripts/blob/0a8c4f764a3aee3b7c1709058dd1488b26bd5f01/utils.py#L180    


    Refector is just to make it a bit more clearer and for me to understand what is going on.
    """

    validation_stats = pd.DataFrame()

    # Check the per-era correlations on the validation set (out of sample)
    validation_correlations = validation_data.groupby("era").apply(lambda d: unif(d["pred"]).corr(d["target"]))

    # mean of correlations
    mean = validation_correlations.mean()

    # std of correlations
    std = validation_correlations.std(ddof=0)

    # sharpe
    sharpe = mean / std

    # store validation score sin validation_stats
    validation_stats.loc["mean", "pred"] = mean
    validation_stats.loc["std", "pred"] = std
    validation_stats.loc["sharpe", "pred"] = sharpe


    # Calculation of APY??
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                    min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    validation_stats.loc["max_drawdown", "pred"] = max_drawdown

    payout_scores = validation_correlations.clip(-0.25, 0.25)
    payout_daily_value = (payout_scores + 1).cumprod()

    apy = (
        (
            (payout_daily_value.dropna().iloc[-1])
            ** (1 / len(payout_scores))
        )
        ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
        - 1
    ) * 100

    validation_stats.loc["apy", "pred"] = apy


    # # Fast mode is ??!
    # if not fast_mode:

    #     # Check the feature exposure of your validation predictions
    #     max_per_era = validation_data.groupby(ERA_COL).apply(
    #         lambda d: d[feature_cols].corrwith(d["pred"]).abs().max())

    #     max_feature_exposure = max_per_era.mean()
    #     validation_stats.loc["max_feature_exposure", "pred"] = max_feature_exposure

    #     # Check feature neutral mean
    #     feature_neutral_mean = get_feature_neutral_mean(validation_data, "pred")
    #     validation_stats.loc["feature_neutral_mean", "pred"] = feature_neutral_mean

    #     # Check top and bottom 200 metrics (TB200)
    #     tb200_validation_correlations = fast_score_by_date(
    #         validation_data,
    #         [pred_col],
    #         TARGET_COL,
    #         tb=200,
    #         era_col=ERA_COL
    #     )

    #     tb200_mean = tb200_validation_correlations.mean()[pred_col]
    #     tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
    #     tb200_sharpe = tb200_mean / tb200_std

    #     validation_stats.loc["tb200_mean", pred_col] = tb200_mean
    #     validation_stats.loc["tb200_std", pred_col] = tb200_std
    #     validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

    # # MMC over validation
    # mmc_scores = []
    # corr_scores = []
    # for _, x in validation_data.groupby(ERA_COL):
    #     series = neutralize_series(unif(x[pred_col]), (x[example_col]))
    #     mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
    #     corr_scores.append(unif(x[pred_col]).corr(x[TARGET_COL]))

    # val_mmc_mean = np.mean(mmc_scores)
    # val_mmc_std = np.std(mmc_scores)
    # corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    # corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

    # validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
    # validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

    # # Check correlation with example predictions
    # per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
    # corr_with_example_preds = per_era_corrs.mean()
    # validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds


    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()