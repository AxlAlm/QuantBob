

#basics 
import numpy as np
import pandas as pd
import scipy



"""
Contains the validation code in https://github.com/numerai/example-scripts/blob/master/utils.py.

I have refactored it just to segment the parts and to understand what is done. 

Some part is are pure copies
"""


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)



def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())


def get_feature_neutral_mean(df, prediction_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          feature_cols)[prediction_col]
    scores = df.groupby("era").apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[TARGET_COL]))).mean()
    return np.mean(scores)


def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized



def apy():
    
    
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

    


def era_correlations(validation_data):
    # Check the per-era correlations on the validation set (out of sample)
    return validation_data.groupby("era").apply(lambda d: unif(d["pred"]).corr(d["target"]))



def mean_std_sharpe(era_correlations):

    # mean of correlations
    mean = era_correlations.mean()

    # std of correlations
    std = era_correlations.std(ddof=0)

    # sharpe
    sharpe = mean / std

    # store validation score sin validation_stats
    df = pd.DataFrame()
    df.loc["mean", "pred"] = mean
    df.loc["std", "pred"] = std
    df.loc["sharpe", "pred"] = sharpe

    return df



def feature_correlation(validation_data, correlations):

    # feature columns
    feature_cols = [c for c in validation_data if c.startswith("feature_")]

    # Check the feature exposure of your validation predictions
    max_per_era = validation_data.groupby("era").apply(
        lambda d: d[feature_cols].corrwith(d["pred"]).abs().max())

    #  max feature exposure
    max_feature_exposure = max_per_era.mean()
    correlations.loc["max_feature_exposure", "pred"] = max_feature_exposure

    # Check feature neutral mean
    feature_neutral_mean = get_feature_neutral_mean(validation_data, "pred")
    correlations.loc["feature_neutral_mean", "pred"] = feature_neutral_mean


    # Check top and bottom 200 metrics (TB200)
    tb200_validation_correlations = fast_score_by_date(
        validation_data,
        [pred_col],
        TARGET_COL,
        tb=200,
        era_col=ERA_COL
    )


    tb200_mean = tb200_validation_correlations.mean()[pred_col]
    tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
    tb200_sharpe = tb200_mean / tb200_std

    validation_stats.loc["tb200_mean", pred_col] = tb200_mean
    validation_stats.loc["tb200_std", pred_col] = tb200_std
    validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

    return validation_stats



def lol():

    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in validation_data.groupby("era"):
        series = neutralize_series(unif(x["pred"]), (x[example_col]))
        mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
        corr_scores.append(unif(x["pred"]).corr(x[TARGET_COL]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

    validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
    validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

    # Check correlation with example predictions
    per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
    corr_with_example_preds = per_era_corrs.mean()
    validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds




def validation_metrics(targets:pd.DataFrame, preds:pd.DataFrame, features:pd.DataFrame = None):
    """
    This is just a refactored version of :

    https://github.com/numerai/example-scripts/blob/0a8c4f764a3aee3b7c1709058dd1488b26bd5f01/utils.py#L180    


    Refector is just to make it a bit more clearer and for me to understand what is going on.
    """


    # Check the per-era correlations on the validation set (out of sample)
    era_correlations = era_correlations(validation_data)
    
    correlations = mean_std_sharpe(era_correlations)


    # apy
    #apy(correlations)


    if features is not None:
        feature_correlation()






    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()