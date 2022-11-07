# Detect counterfeit banknotes with logistic regression

### ✅ Business Case

In this project, we are creating a model to detect fake banknotes using a Logistic Regression algorithm and labelled data containing the characteristics of genuine banknotes. 


### 📉 Dataset

This project was completed as a part of the [Data Analyst nanodegree](https://openclassrooms.com/en/dashboard/paths) displayed by Openclassrooms. 
The dataset is therefore provided by the platform as a part of the program.

### 📒 Repository
The repository is available on [GitHub](https://github.com/AurelieGIRAUD/Data_Science_Projects/tree/main/Logistic_Regression). It describes how to apply a Logistic Regression Model using the following steps:

1. Exploratory Data Analysis (EDA)
  
  - Univariate and Bivariate Analysis. 
  - Handling outliers
  - Data Visualizations supported by MatplotLib.
  
 2. Principal Component Analysis (PCA)
  
  - Scree Plot Analysis.
  - Loadings Plots.
  - Scores Plots.
  - Selection of the principal components.
  
 3. Logistic Regression
  
  - Train and Apply a logistic regression model to detect fake banknotes using their dimensions.
  - Performances Analysis.


### 🎯 Key take-away


**We start the analytic by evaluating the diversity of income across a selection of 5 countries, using the Lorenz curve and Gini Index.** 


✅ The following distribution highlight the broad range of incomes among the selection: the average income move from left to right (from low to high) depending on the country. Georgia and El Salvador are the 2 countries with the lowest average income in the given selection.

<img src="images/rsz_1screenshot_2022-11-06_at_210543.png"/>

1. Lorenz curve

The Lorenz concentration curve is a way to represent the distribution function of a variable X. It is used in particular in economics to measure inequalities in the possession of wealth. We will use it here to assess inequalities in terms of the distribution of incomes among the group of countries selected. The furthest the Lorenz curve is from the first bisector line, the more inequalities of revenue in the given country.


```python
def lorenz_curve(X):
    lorenz = np.cumsum(np.sort(X['income'])) / X['income'].sum()
    lorenz = np.append([0],lorenz) 
``` 

<img src="images/rsz_screenshot_2022-11-06_at_211118.png"/>

✅ In the example above, we can see here that there are more inequalities in Argentina than in Denmark.

2. Gini Index

The Gini index represents the area between the Lorenz curve and the first bisector line. It is a good metric to summarize the information from the Lorenz curve. The smallest the Gini Index, the least inequalities in the given country. 


<img src="images/rsz_11screenshot_2022-11-06_at_211632.png"/>


✅ In the example above, we can see that Denmark and Iceland are the countries with the least inequalities in our given selection. 


3. Conditionnal probabilities

At that stage, we have two of the three desired explanatory variables: the average income and the gini index for each given country. 
We only lack now the income class of an individual's parents to move forward into the prediction. The idea is to use this variable to predict the probability for an individual to fall into the same lass of income than its parents - this will bring us closer to predict the revenue of an individual.

We suppose here that we associate with each individual a unique class regardless of the number of parents. We are going to simulate this information using a coefficient (specific to each country) measuring a correlation between the income of an individual and the income of its parents. This coefficient will be called the elasticity coefficient or relative IGM in income. It measures the intergenerational income mobility.



  

  
