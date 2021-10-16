# QuantBob

QuantBob is my pipeline for developing models for [Numerai](https://numer.ai/).

---

## Data Overiew


### Example

Data will look like this:


 ID  |   era |  feature_1 | ... | feature_n | target | target_<TYPE 0>_20 | target_<TYPE 0>_60 | ... | target_<TYPE 1>_20 | target_<TYPE 1>_60 | 
---- |------ | ---------- | --- | --------- | ------ | ------------------ | ------------------ | --- | ------------------ | ------------------ | 
 z   |     1 |         0  | ... |  0.23     | 0.5    |              0.5   |              0.11  | ... |             0.5434 |               0.31 |
 x   |     1 |         0  | ... |  0.30     | 0.75   |              0.75  |              0.11  | ... |             0.5434 |               0.31 |
 y   |     2 |         1  | ... |  0.78     | 0.25   |              0.25  |              0.11  | ... |             0.5434 |               0.31 |



#### ID

simply a id for a unique stock. This ID is unique for each ERA so you cannot track a stock through time. (I.e. we are not doing time series prediction)


Some good forum post to clear some confusion

https://forum.numer.ai/t/noob-question-regarding-data/1700


#### era 

Eras are in 5 week intervals, and they seem to overlap by rolling. E.g. if we have 20 days era 1 is 1-5, era 2 2-6, ..., era 5 5-9

#### features

These are the features we need to work with. These are "anonymised" meaning we have no idea what they represent.

There are 1050 features and no groups.

#### target

"The target represents an abstract measure of performance ~4 weeks into the future." - [numerai](https://docs.numer.ai/tournament/learn)


There are 10 different types of targets and 20 targets in total. However, we are only scores on one target. This currently "target_nomi_20" (e.g. "target_<TYPE 0>_20" )

The number following some targets are the number of days in the future?

---

## Task

The task is to:

    Given a set of feature representing a sample, predict one out out 5 discrete values


main target are the following discrete values {0, 0.25, 0.5, 0.75, 1}. Other tatgets might have other values
    
What we are essentially doing is predicting the future behavior of a stock; given some state of a stock at time t, we try to predict something about its stock at time t+n, where n is some period of time

#### What about the other 19 targets, what should we do with these? 

these targets are referd to as Auxillary Targets. These can be used to train models and its apparently good to use them to train models, as models seems to generalize better if trained on these as well (see [ref](https://github.com/numerai/example-scripts/blob/master/analysis_and_tips.ipynb))

--

## Evaluation

All models are scored based on the rank-correlation (spearman) with the target

---

## what about ERAS?

We do have more than just features; we also have eras (time). Even though eras do not change the nature of the task, eras can be used in various ways. E.g. creating ensambles, train a model per era.

---

## Data Analyis

Currently I have just summerized the points made [here](https://github.com/numerai/example-scripts/blob/master/analysis_and_tips.ipynb):

- Each Era has around 4000-5000 rows

- Target Distribution (not the same for all targets):
    5% of target are 0
    20% of target are 0.25
    20% of target are 0.75
    50% of target are 0.5
    5% of target are 1

- some features are very correlated

- feature correlation change over time

- some features are predictive on their own ( what does this mean??)

- Feature exposeure of different target can be different

- ERAS overlap

- ERAS are homogenous but different from eachother (i.e. using overlapping eras or all of them will not be so different. But using all eras might increase overfitting)

- signal-to-noise ratio is low (????)

- results are sensitive to choice of paramaters

- cross validation is good (?)

- Models with large exposure to individual features tend to perform poorly or inconsitently out of sample

- Training with the auxillary target can result in models with different pattersn of feature exposure


---

## Overview of QuantBob

QuantBob is a very simple library aimed to do four things:

1) download and upload numerai data
2) train and predict with models
3) evaluate models
4) document and store the score of various models


We have two main classes:

- Dataset: will download and upload data.

- Model: will contain everything from feature selection, data splitting/ensamble configuration and model training and prediction

Code used for evaluation is copied from the numerai/example-scripts repo and slightly refactored (just to for me to understand that it does).


Example:

```python

#sklearn
from sklearn.ensemble import GradientBoostingRegressor

# quantbot
from quantbob import Dataset
from quantbob.models import BasicEnsambler
from quantbob.evaluation import validation_metrics

# download the dataset
dataset = Dataset()

# we select a model. BasicEnsambler is a naive model which takes a sklearn ensamble classifier
# and trains it on all the data. I.e. it doenst do any type of features selection, normalisation, ensambling
# by era or anything. 
model = BasicEnsambler(GradientBoostingRegressor, n_estimators=2, learning_rate=0.1, max_depth=1, random_state=0)

# Fit the model
model.fit(dataset.train_data)

# predict on the validation set
dataset.val_data.loc[:, "pred"] = model.predict(dataset.val_data)


# lastly, we evaluate our model by using the validation metric given by Numerai.
validation_metrics(dataset.val_data)
```

