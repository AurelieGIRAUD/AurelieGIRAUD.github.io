# Multiple Linear_Regression applied to a Marketing Case Study

## Context

Most firms that think they want advanced AI/ML really just need linear regression on cleaned-up data [Robin Hanson]

This repository describes how to apply a Multiple Linear Regression Model using Scipy and how to perform regression diagnosis to tackle uncertainties. 
The logic and method are detailled in the following article: https://towardsdatascience.com/perform-regression-diagnostics-and-tackle-uncertainties-of-linear-models-1372a03b1f56 

## Dataset

The dataset can be found here: https://www.kaggle.com/fayejavad/marketing-linear-multiple-regression
It describes the advertising experiment between Social Media Budget and Sales (in Thousands $ ) and hold 200 experiments.

## The 4 peace keeper

There are four principal assumptions which support using a linear regression model for the purpose of inference or prediction:

1. Linearity: 
We must have a linear relationship between our features and responses. This is required for our estimator and predictions to be unbiased.
<img src="images/Screenshot 2022-10-21 at 18.18.50.png"/>

The next ones are concerning the residual:

2. Normality:
Residuals must be Normally distributed (i.e variance tend to 1 and mean tend to zero). This is necessary for a range of statistical tests, such as the t-test. We can relax this assumption in large samples due to the central limit theorem.
<img src="Screenshot 2022-10-21 at 18.19.12.png"/>

3. Homoscedasticity:
Means that the residuals have constant variance no matter the level of the dependent variable.


4. Independence:
Residuals must be totally free of autocorrelation.


## The 5th trouble maker

For multiple linear regression, you also need to check:

5. the Absence of multicollinearity:
Multicollinearity refers to when two predictors (or more) are providing the same information about the response variable. This can be a problem for the model as it generates:

- Redundancy → leading to unreliable coefficients of the predictors (especially for linear models)

- High variance of the estimator → tends to overfit meaning that the algorithm tends to model the random noise in the training data, rather than the intended output.

- An important predictor can become unimportant.


