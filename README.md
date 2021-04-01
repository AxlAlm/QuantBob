# QuantBob



#### Data Analysis


Data will look like this:


 ID  |   era |  feature_1 | ... | feature_n | target |
-----+-------+------------+-----+-----------+--------
 z   |     1 |         0  | ... |  0.23     | 0.5    |
 x   |     1 |         0  | ... |  0.30     | 0.75   |
 y   |     2 |         1  | ... |  0.78     | 0.25   |


##### ID

simply a id for a unique stock


##### era 

Eras tells us which time period/point the row refers to. Each unique row refers to a stock at some point in time.
The intervalls seem to be months and are sequential

https://forum.numer.ai/t/relation-of-eras-with-time-periods/371

##### feature_X

These are the features we need to work with. These are "anonymised" meaning we have no idea what they represent.


##### target

Target is t = t E {0.0, 0.25, 0.50, 0.75, 1}


Some good forum post to clear some confusion

https://forum.numer.ai/t/noob-question-regarding-data/1700



#### Task


The task is simply to **for each row predict the target given the features**.

The eras are slightly confusion as they give you the idea of a time series task it really not as we do not have any idea of how 
a stock (row) develop through time as each row has a unique id.

In normal time-series task we have the following data:

    xt, xt+1, ... , xt+n


e.g,

    Tesla_stock_pricet, Tesla_stock_pricet+1, ... , Tesla_stock_pricet+n


or
    wordt, wordt+1, ... , wordt+n
    (
        Time series in NLP might be better described as a sentence uttered through time and the TIMESTEPS are words
        
        sentencet, sentencet+1, ... , sentencet+n

        each t is a word idx
    )


But in this dataset we cannot create time lines like this as we do not have anything to track a stock over time. We only 
have:

        collection_of_stockst, collections_of_stockst+1, .... , collections_of_stockst+1

where we have no idea if 

    collection_of_stockst containts stocks in collections_of_stockst+1, or if so which they are.



##### Task Modeling

Here are some possible ways to model the task
 
####  PAST vs FUTURE ( what ever any of these are )

Train on the past, ignore eras and predict on the future


#### 