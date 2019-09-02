# Kaggle_Santander_Contest
Classification problem on imblanaced dataset
Dataset:
  - 200 variables (each having approx. gaussian distribution)
  - binary target variable (only 14% being labeled as '1')
  - 200 000 observations

Implementation:
1. Creating synthetic variables to augment number of '1' thus avoiding problem of imbalance.
2. TrainTest Split, 0.2 - test size.
3. Choosing LightGBM model hyperparams (trail and error method combined with GridSearch)
4. Training LightGBM model. 
5. Evaluating model with Confusion Matrix (sensitivity), Accuracy and ROC Curve. 
6. Exporting predictions. 

Final result on Kaggle (area under ROC): 0.897 (top50%). 
  
