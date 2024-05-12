# Report: Predict Bike Sharing Demand with AutoGluon Solution
Tharusha Dilhara Jayadeera

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When I first attempted to submit my predictions on Kaggle, I encountered an issue. The forecasted bike counts included negative numbers, which were not permissible. To resolve this, I had to adjust the output of the predictor to guarantee that all predicted values were non-negative.

### What was the top ranked model that performed?
WeightedEnsemble_L3 was the best performing model in the first training cycle.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?

It is probable that the Exploratory Data Analysis (EDA) entailed employing descriptive statistics or histograms to analyze the feature distribution. I've located features that might be outliers or have skewed distributions.

1. I added a new functionality by dividing the date time column into     
  sections for hour, day of week (date), month and year.

    train["year"] = train["datetime"].dt.year
    train["month"] = train["datetime"].dt.month
    train["day"] = train["datetime"].dt.dayofweek
    train["hour"] = train["datetime"].dt.hour
    train.drop(["datetime"], axis=1, inplace=True)

2. Feature types: To make sure AutoGluon handles categorical features like 
   "season" and "weather" properly, I changed them to categorical data types using astype("category").

   train["season"] = train["season"].astype("category")
   train["weather"] = train["weather"].astype("category")
   test["season"] = test["season"].astype("category")
   test["weather"] = test["weather"].astype("category")

### How much better did your model preform after adding additional features and why do you think that is?

Bike sharing demand may have shown temporal trends if features like hour, day, and month had been included. For instance, there may be a spike in bike sharing on particular days or hours of the week. With the help of these extra variables, the model had more data to work with and could predict outcomes more precisely.

Thus, the initial attempt, devoid of any extra characteristics, was 1.80711. It decreased to 0.46556 with the features added, indicating a roughly 75% improvement in accuracy.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tweaking probably involved changing variables such as the number of epochs, learning rate, or trees in a GBM model. The model may have been better able to learn from the data and produce accurate predictions if these hyperparameters had been optimized.

Therefore, 0.46556 was the effort with extra characteristics. The accuracy decreased by about 4% to 0.48177 after the addition of hyperparameters. I made several attempts to reduce the score compared to the more features attempt by adjusting the parameters, but I was unsuccessful. My attempts and the accompanying scores are shown below.

    fileName                     date                 description                        status    publicScore  privateScore  
    ---------------------------  -------------------  ---------------------------------  --------  -----------  ------------  
    submission_new_hpo.csv       2024-05-09 21:57:26  new features with hyperparameters  complete  0.48177      0.48177       
    submission_new_hpo.csv       2024-05-09 21:45:15  new features with hyperparameters  complete  0.47778      0.47778       
    submission_new_hpo.csv       2024-05-09 21:35:53  new features with hyperparameters  complete  0.47536      0.47536       
    submission_new_hpo.csv       2024-05-09 21:24:09  new features with hyperparameters  complete  0.47009      0.47009       
    submission_new_features.csv  2024-05-09 21:17:34  new features                       complete  0.46276      0.46276       
    submission.csv               2024-05-09 21:05:54  first raw submission               complete  1.80641      1.80641       
    submission_new_hpo.csv       2024-05-09 19:38:51  new features with hyperparameters  complete  0.48177      0.48177       
    submission_new_features.csv  2024-05-09 19:19:25  new features                       complete  0.46556      0.46556       
    submission.csv               2024-05-09 18:59:35  first raw submission               complete  1.80711      1.80711


Conversely, when compared to the initial training, hyperparameters attempt is more than ~74% accurate.

### If you were given more time with this dataset, where do you think you would spend more time?
Given that my attempt at lowering the hyperparameters score was unsuccessful, I would work with the hyperparameters further to improve the accuracy of the model.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|train_data|eval_metric|presets|-53.103764|
|add_features|train_new_more_fts|eval_metric|presets|-33.981768|
|hpo|GBM|NN_TORCH|NN_TORCH|-38.496203|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![image](https://github.com/it22606006/Predict-Bike-Sharing-Demand-with-autoGluon-tdjayadeera/assets/128974370/df77bddb-e168-47cc-a7bd-d4e462eeddc9)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![image](https://github.com/it22606006/Predict-Bike-Sharing-Demand-with-autoGluon-tdjayadeera/assets/128974370/18e44389-a2c0-404f-a48e-4428d6dc324f)


## Summary
This paper investigated the use of AutoGluon for demand prediction in bike sharing systems. The model's high RMSE score (1.80711) at first was mostly caused by the fact that it did not limit outputs to non-negative values. The accuracy of the model was greatly increased by incorporating features that recorded various variables of time, such as the hour, day of the week, and month, resulting in a reduced RMSE of 0.46556. With an RMSE of 0.48177, subsequent attempts to further refine the model through hyperparameter modifications only slightly improved performance, suggesting potential problems with overfitting or less-than-ideal hyperparameter settings. In the future, experimenting with other hyperparameter tuning strategies and using regularization techniques might aid in improving accuracy.
