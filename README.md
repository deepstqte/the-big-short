# Credit Risk Report

By [Pierre-Olivier Bonin](https://github.com/pierreolivierbonin) & [Hamza Benhmani](https://github.com/gzork).

Project for [CEBD1260 - Introduction to Machine Learning](https://www.concordia.ca/cce/courses/details.html?subject=CEBD&catalog_number=1260)

## Executive Summary

This report summarizes a machine learning project we have conducted using data from Ant Finance’s client’s historical records. Based on seven different datasets, we implement a comprehensive machine learning pipeline that allows to predict which clients are likely to default loan payment. We train a benchmark LightGBM model, aggregate features from the different datasets, merge them into the main one, build and train a new LightGBM model, compare the performance of the models, and lay out the most important features that predict payment defaulting. The improved model yields an AUC score of 0.785900 for correctly predicted payment defaults (a.k.a. "1"s), an improvement of about 2% over the benchmark model. Some of the most interesting features among those with the highest predictive power include the age of the client, the number of consecutive days they have been employed, as well as a number of features we have engineered based on the datasets. This highlights the importance of feature engineering as part of machine learning projects.

## Context and Problem Statement

Ant Finance is a Chinese financial corporation that specializes in lending money to individuals with a risky credit profile, specifically those with little to nonexistent credit history. Ant Finance operates globally and has over 1.2 billion active customers, most of whom are located in Asia and China. This means that the firm has been able to collect enormous amounts of data over the years. This report on credit risk proposes an application of machine learning (ML) to predict the likelihood of defaulting on loan payments. To that end, seven interrelated datasets based on the characteristics of Ant Finance’s clients’ historical records are going to be used. The datasets are presented and summarized as follows.

### The Datasets Summary

| File Name | Description |
| -------- | -------- |
| application_{train\|test}.csv | "The client application for credit currently, which contains current application information and TARGET (how difficult the client can repay the loan)." |
| bureau.csv | "The previous application of a client in other financial institutes that were reported to Credit Bureau (for clients who have a loan in Ant finance). For every loan in these samples, there are as many as number of credits the client had in Credit Bureau before the application date." |
| bureau_balance.csv | "Monthly balances of previous credits in Credit Bureau. This table has one row for each month of history of every precious credit reported to Credit Bureau— i.e. the table has number of loans in sample * number of relative previous credits * number of months where Home Credit have some history observable for the previous credits." |
| previous_application.csv | "Monthly balance snapshots of precious POS (point of sales) and cash loans that the applicant had with Home Credit." |
| POS_CASH_balance.csv | "Monthly balance snapshots of previous credit cards that the applicant has with Home Credit." |
| instalments_payments.csv | "All previous applications for Home Credit loans of clients who have loans in Home Credits’ samples." |
| credit_card_balance.csv | "Repayment history for the previously disbursed credits in Home Credit related to the loans in current samples." |

The descriptions are quoted from: Ant Finance. n.d. “Customer credit risk analysis.” (File name: “description.docx”, provided by the course's instructor). Unpublished.

## Designing a Machine Learning Dash App to Model Credit Risk

Our Approach to Preprocessing and Feature Engineering:

To construct a machine learning model based on the datasets presented above, the data first need to be preprocessed, and then features need to be extracted from the data. Specifically, for each dataset, missing values need to be evaluated and filled using appropriate algorithms and methods. Then, categorical features need to be encoded. Encoding facilitates ML modelling because the available algorithms often cannot handle categorical features on their own. Two encoding methods have been used for the current use case: one-hot encoding, and label encoding. One-hot encoding creates dummy variables out of each category within a categorical feature (or variable). In other words, each category within that feature is transformed into a new, binary `[0, 1]` column. The one-hot encoding method we have implemented in our ML pipeline for this report is based on an implementation of [`pandas.get_dummies` function](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html). As for the second, alternative encoding method, label encoding assigns integers to each category within a feature. For this method, our pipeline implemented [Scikit-learn’s LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

For all of the above as well as for imputing missing values, we chose to design a single script that includes custom-made Python functions for each preprocessing and feature engineering task to be completed. This script, called [`utils.py`](https://github.com/gzork/the-big-short/blob/main/utils.py), thus proposes eight functions to deal with a series of tasks to be completed. These eight functions are described [in this table](#Custom-helper-functions).

As part of our ML pipeline, we have designed Dash application that allows the user to experiment with feature engineering by directly tweaking the missing value imputation method, the aggregated features included, as well as the threshold following which categorical features are one-hot-encoded. The application is available [here](https://the-big-short.gzork.com/) and works as follows. For each data subset, the user may enter `1` in each cell where they wish for a specific feature to be included. They then select a method to fill missing value (either `mean` or `median`), and they may select a desired threshold for the one-hot encoding, as already described above. They can then produce a main dataframe by clicking on a button of the same name. Then, bearing in mind that the model underlying this application is [LightGBM](https://lightgbm.readthedocs.io/en/latest/), the user scrolls down and is able to experiment with hyper-parameter tuning directly with the application. Configurable settings include the following:

- early-stopping rounds
- number of parallel threads
- test size
- learning rate
- number of leaves
- feature fraction
- bagging frequency
- bagging fraction
- minimum data in each leaf.

### Custom helper functions

| Function Name | Short Description |
| -------- | -------- |
| na_catfiller(df) | This function identifies categorical and Boolean features, then replaces missing values with "missing_data" value. |
| na_numfiller(df, aggregation_func="mean") | This function identifies non-categorical features, then identifies non-Boolean features within the resulting list of features, and then identifies features with missing values within the resulting list of features. It then returns the list of features identified. Lastly, it fills missing values identified within those features with the method selected by the user (either mean or median values). |
| str_one_hot_encoder(df, unique_threshold=10) | By contrast with the other encoding function below, this function applies a straightforward one-hot encoding technique, using a threshold of 10 |
| aggr_dicts_json_to_dict(data_from_json) | This function transforms a json aggregation dictionary to a regular dictionary. |
| str_cat_encoder(df, method_switch=10) | This function applies one-hot encoding or label encoding on categorical string colmns based on the number of unique values in the given dataframe. One-hot encoding is used if thenumber of unique values is smaller or equal to the "method_switch" threshold, and label encoding is used if bigger. |
| merge_with_aggr(main_df, secondary_df, fk_columns, aggr_dict, column_prefix) | This function merges two dataframes based on a key column, an aggregation dictionary, and assigns a prefix to the new columns names created. |
| optimize_inttypes(dataframe, specify="auto") | This function optimizes memory usage by downgrading data types. Selects the `integer` type columns in the dataframe. If not specified, automatically optimizes them by downgrading to the smallest possible float type (e.g. `int8`). If specifying, user may select `int32`. |
| optimize_floattypes(dataframe, specify="auto") | This function optimizes memory usage by downgrading data types. Selects the `float` type columns in the dataframe. If not specified, automatically optimizes them by downgrading to the smallest possible float type (e.g. `float8`). If specifying, user may select `float32`. |

## Results, Model Evaluation and Feature Importance

This section draws mainly from our Jupyter notebook, which can be consulted [here](https://github.com/gzork/the-big-short/blob/main/Full_ML_pipeline.ipynb). To get a baseline over which to compare our ML model once it will be trained and tuned, we first build a benchmark model based on the application_train.csv dataset. To do so, we define a `train_model()` function that applies a one-shot train-test split and fit a LightGBM model to our dataframe, and then train the model. We fill missing values and encode appropriate columns, and then train the model using our custom function `train_model()`. Results are as follows:

```
Early stopping, best iteration is:
[1560] training's auc: 0.834241 training's binary_logloss: 0.222639 valid_1's auc: 0.765379 valid_1's binary_logloss: 0.244857
```
**valid_1's auc: 0.765379**

We find that the benchmark model is already quite good, with a valid 1’ AUC score of 76.5%, indicating that the model correctly predicted 76.5% of the clients who defaulted their loan payments.

Next, we try to improve this performance. We use a dictionary object to apply our `na_catfiller`, `str_one_hot_encoder`, and `na_numfiller` functions for the aggregations for each dataset that will ultimately be merged in the main dataframe. We deal with missing values, and then we finally have a single, main dataframe on which to build our model. We first conduct a one-shot train- test split approach, train, and run the model. Results are as follows:

```
Early stopping, best iteration is:
[2615] training's auc: 0.913419 training's binary_logloss: 0.186508 valid_1's auc: 0.786341 valid_1's binary_logloss: 0.237927
```
**valid_1's auc: 0.786341**

Results show a significant improvement, with over 2% more correctly predicted cases. We continue experimenting with our features and increase the threshold for one-hot encoding to fifty. This will encode more categorical variables into binary (one-hot-encoded) variables. We train and run our model. Results are as follows:

```
Early stopping, best iteration is:
[2689] training's auc: 0.916154 training's binary_logloss: 0.185061 valid_1's auc: 0.78624 valid_1's binary_logloss: 0.237854
```

**valid_1's auc: 0.78624**

This time, the performance very slightly deteriorated, but the change is not significant, so we move on and run our model using a KFold cross-validation technique, where `k = 5`. Results show a minimum valid-1’s AUC score of **0.779807**, and a maximum of **0.789839**. Overall, the average performance of our model over all of the five folds is **0.785900**. This represents a 0.020521 difference, or about **2.1%** in performance with the benchmark model, which is a satisfactory improvement in the context of this use case.

As a last step, we look at the importance of the features driving the predictions made by our model. It seems that features from several different datasets have been determining factors in predicting whether a client is likely to default loan payment, which highlights the importance of taking full advantage of data from multiple sources when analyzing big data-driven problems.

![image](https://user-images.githubusercontent.com/7915931/112740101-2d93a480-8f48-11eb-9924-5b04976c6e70.png)

Using the HomeCredit file that describes the features of the datasets we have been using for this project, we can explain the meaning of the ten most important features that determine the likelihood of loan payment defaulting. The three features with the highest impact — `EXT_SOURCE_2`, `EXT_SOURCE_1`, and `EXT_SOURCE_3` — are described as the "normalized score from [an] external source". Those sources will remain unknown to us. The next one, `DAYS_BIRTH`, shows the client’s age counted in days; `AMT_ANNUITY` refers to the annuity paid to the credit bureau, which generally means a fixed sum of money paid on a regular basis; `AMT_CREDIT` refers to the "final credit amount on the previous application" of the client; `installments_payments_AMT_PAYMENT_min` refers to one of the features we have engineered, which represents the minimal amount paid by the client on previous credit on a given installment; `DAYS_EMPLOYED`, although self-explanatory, refers more specifically to the number of "days before the application the person started current employment"; `AMT_GOODS_PRICE` is described as the "price of good that [the] client asked for (if applicable) on the previous application"; and the tenth most important feature, `DAYS_ID_PUBLISH` has to do with time relatively to the application date, described as the number of “days before the application [the] client change[d] the identity document with which he applied for the loan”.

Out of the ten remaining most important features of our “Top 20”, eight are features we have engineered, which is a pretty interesting finding. To be clear, such a finding highlights the importance of spending time on feature engineering for machine learning projects. In the current case, a total of 9/20 of the most importance features of our model have been the product of our efforts with transforming and making sense out of the data. This has allowed an improvement in performance of about 2% in comparison with the benchmark model. And even if, in a different scenario, it did not yield significant improvements for the model’s performance, the features we have engineered have allowed to narrow down the variables that are the most apt at predicting loan payment defaulting, which should be considered valuable in and of itself.
