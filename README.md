# RegressionEnsemble
This repo contains notebook with multiple models built for regression.
Following models were evaluated for this regression task:
1. Linear Regression
2. Lasso Linear Regression with Cross Validation
3. Random Forest
4. Gradient Boosting
5. XgBoost 

Comparison of model performances showed XgBoost did marginally better especially over gradient boosting. 

Also implemented a python script to automate predictions using xgboost model. 

As part of preprocessing :
1. Implemented feature selection using :
	a. Standard deviation 
	b. Multi colinearity
	c. Missing value % 
2. Min max scaler 

Also sclaed target variable to lie between 0-1 so that gradient descent would converge quickly. 

All of these values are stored as pickle files to be used later for automating entire process. 


