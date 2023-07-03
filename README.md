# American-Express---Default-Prediction
# Dataset Description
The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.

The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:
D_* = Delinquency variables
S_* = Spend variables
P_* = Payment variables
B_* = Balance variables
R_* = Risk variables
with the following features being categorical:

['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
Your task is to predict, for each customer_ID, the probability of a future payment default (target = 1).

Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.

Files
train_data.csv - training data with multiple statement dates per customer_ID
train_labels.csv - target label for each customer_ID
test_data.csv - corresponding test data; your objective is to predict the target label for each customer_ID
sample_submission.csv - a sample submission file in the correct format

# Modeling
The first idea for data modeling is to keep only the last record for each customer. I implemented this in the code, the next step is to remove the columns where there are more than 80% omissions, and fill the rest with the mean and mode for the categorical variables, I perform these steps for both the training table and the test table. The next step is to get rid of collinearity using the correlation matrix. We delete columns with a correlation value greater than 0.9

Having processed the data in this way, I proceed to modeling. I train the obtained dataset using the lightgbm model with the following parameters:
{'objective': 'binary','n_estimators': 1200,'metric': 'binary_logloss','boosting':
'gbdt','num_leaves': 90,'reg_lambda' : 50,'colsample_bytree': 0.19,'learning_rate':
0.03,'min_child_samples': 2400,'max_bins': 511,'seed': 42,'verbose': -1}

Accuracy calculation

The evaluation metric 𝑀 for this problem is the average of two measures: the normalized Gini coefficient 𝐺 and the default coefficient at 4%, 𝐷

𝑀=0.5⋅(𝐺+𝐷)
The default metric obtained at 4% is the percentage of positive labels (default) captured within the top 4% of predictions and represents the sensitivity/recall statistic.

For both 𝐺 and 𝐷, negative labels are given a weight of 20 to adjust for downsampling.

This indicator has a maximum value of 1.0.

The Gini coefficient is a popular metric on Kaggle, especially for unbalanced class values.
It can be considered that the Gini estimate is simply a reformulation of the AUC:
𝑔𝑖𝑛𝑖=2×𝐴𝑈𝐶−1
As for why to use this instead of the commonly used AUC, is that a random forecast will give a Gini score of 0 as opposed to AUC which is 0.5.
It is also worth mentioning what AUC is

An ROC curve is a graph that shows the performance of a classification model
for all classification thresholds. This curve displays two parameters:
True Positive Rate
False Positive Rate
AUC stands for “area under the ROC curve”. That is, AUC measures the entire two-dimensional area under the entire ROC curve from (0,0) to (1,1).

The result is not so bad, considering that the dataset contained a very large number of gaps. Also, the very approach of data aggregation is very risky, leaving only the last record of the client, we could lose useful information. Let's try to aggregate data in a different way this time - aggregate all numeric columns by customer_id as sum, standard deviation, minimum, maximum, last, first records; categorical variables are aggregated as number, first, last, number of uniques, mode.
In this way, we transform our dataset so that each customer corresponds to one row, without losing important information from past records.
