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

QuantBob is a pipeline for testing, evaluating and keeping track of models.

It consist of the following modular components:

- Dataset
    - downloads and stores the Numerai data

- DataSampler
    - a data sample will contain techniques and approach to create data splits. This can include creating various dataset splits for each era, random sampling features etc etc
    - this class will retuns a iterator of dataset to train models for
    - each dataset will have X

- Model
    - model is the model used to train the data. This can be any sklearn, xgboost or whatever algorithm you want.

- Evaluater
    - evaluation will evaluate all the 

- QuantBob
    - For Bob to to his job he needs a Dataset, Datasampler, Model and an Evaluator.


Example:

```python
from quantbob import QuantBob
from quantbob import Dataset
from quantbob import Evaluator
from quantbob.data_samplers import TestDatasampler
from lightgbm import LGBMRegressor

# Init Bob
bob = QuantBob(
                dataset = Dataset,
                evaluator = Evaluator,
                data_sampler = TestDatasampler,
                model = LGBMRegressor
                )

# Train bob
bob.train()

# gives us the scores
bob.scores()

# Predict with bob, return path to csv predictions
pred_csv = bob.predict(pred_Data)
```

