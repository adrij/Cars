# Analysis of the second-hand car market in Hungary

I am thinking of buying a second-hand car for my mother. My only problem is, I don't really known a lot about them. There are couple of trade-offs to consider when someone is buying a used car:
- Which car is supposed to be cheaper, an older one with less kilometer, or a newer one with more?
- What are the most important features, that I have to consider?

I decided to come up with a statistical model, which defines the relationship between the relevant features of the car and the price. Or at least gives me a sign, whether a car is overpriced.

## Goals

Scrape second-hand cars for sale in the hungarian market from the web and create a model to predict the valid price of the cars based on its relevant features.

## Approach
### _Step 1: Organizing the Data_

Statistical data was gathered from the webpage of second-hand-cars for sale (www.hasznaltauto.hu). At first all the links of the available cars were collected. By means of the collected links, all the available features of the cars were imported into a dataframe. The BeautifulSoup Python library was utilized to develop the scraping tools.
After removing the duplicates, the raw dataset consisted of 90.874 cars, which were analysed and cleaned during the EDA (explorative data analysis) part.

Based on the cars for sale on the hungarian second hand market, the most common car type for sale is OPEL ASTRA. The most popular car types can be seen in the following graph.

![Topcars](/pictures/Top16cartypes.png)

### _Step 2: Data Exploration_

All raw features were transformed into numerical values. The non-relevant information, which cannot be used for modelling, was dropped from the database. During the univariate analysis of the features, the missing values and outliers were also treated. The categorical variables were transformed into numerical by means of dummy variables. Finally the following variables were used in the analysis:

1. Variables from the webpage:
* Nr_Doors: nr of the doors in the car
* Luggagerack_final: Size of the luggage rack
* Capacity_final: Capacity
* WeightOwn_final: Own weight
* WeightTotal_final: Total weight of the car
* Speedometer_final: Speedometer
* Nr_Person_gr: The number of person was grouped into 3 bins
* Rating_gr: rating based on users of the website
* Flag for Guarantee: if a car still has a guarantee or not
* Performance_LE_final: Performance in LE
* Performance_kW_final: Performance in kW
* Age_year_final: Age in years
* Age_final: Age in days
* Flag_first_w: Flag for first wheel drive
* Flag_back_w: Flag for back wheel drive
* Flag_diesel: Flag whether the car needs diesel as fuel
* Flag_benzin:Flag whether the car needs benzin as fuel
* Flag_electro: Flag, whether it is an electrocar

2. Variable - Car classes 
  A new dataset was imported from a German webpage to define car classes. The dataset contained car classification categories based on producer and type. These new categories with their average prices can be seen in the following table:

![Avgprice_carclass](/pictures/Averge price of car classes.png)

3. Variable - Producers
  There are in total 66 car producers in the dataset. I discarded producers with less than 100 cars for sale, since they wouldn't have much predictive value. The dummies for the remaining producers resulted in an additional 40 features.
The average prices for the top producers can be seen in the following table:

![Avgprice_producer](/pictures/Averge price of top producers.png)
  
  The red columns show the distributions: the most common producers are OPEL, VOLKSWAGEN, FORD and BMW. The blue line corresponds to the     average price: the most expensive producers on average are BMW, AUDI and MERCEDES-BENZ. As seen before, there is an essential             difference between the price of car classes. It is also helpful to illustrate the stacked distribution of the cars based on the classes:

![Stacked distribution](/pictures/stacked_dist.png)

  Among the more expensive cars, the ratio of the large family cars and executive cars is much higher than among the cheaper ones. The c     cheaper producers sell mainly mini and small cars.

4. Variable - Average price of the car type
  I also wanted to involve the car type in the database: I calculated the average price of a car type and added it as a model variable.
  Finally, the dataset consisted of 84.453 observations with 62 input and 1 target variable. After scaling the input variables I started     the multivariate analysis by trying out different statistical models to predict the price of a specific car.

### _Step 3: Multivariate Analysis_

As seen in the heatmap below, some variables are correlated, so I needed to exclude the following variables:  
* Performance_LE_final
* WeightOwn_final 
* Flag_benzin
* Flag_first_w
* Age_final

![Heatmap](/pictures/heatmap.png)

I used the StandardScaler function to scale the dataset, since the features had different magnitudes. I split the dataset into training and validation sample: I will train the model on the training dataset and check the performance of the prediction on the validation sample.

I executed different regression models, decision trees, random forests, support vector machines and boosting techniques in order to find the best statistical model to predict the car prices. I used the R2 score and mean squared error(MSE) of the training and validation set to compare the models regarding the performance and overfitting.

#### 1. Linear Regression models:

In linear regression models the target variable is expected to be a linear combination of the input features. I fitted the following regression models:

  1. Linear regression with the first k best features: The best features were selected based on a univariate regression test called F Regression, which computes the correlations between each features and the target variable. 
If all the variables were chosen (k=number of features), the R2 score for the training and validation dataset is 75,62% and 75,28% respectively. Based on the number of features (k), the R2 score has the following values:

![Linreg](/pictures/select_k_linreg.png)

  As seen on the graph, after k=10 variables there is not much difference in the R2 score.  The performance of the training and validation   dataset are very similar, there is no sign of overfitting.
  
  2. I also fitted polynomial regression on the data: I tested the polynomials till the d=4th degree.  I used a pipeline to 
* select the first 20 best features
* transform these features into polynomials
* fit a linear regression model on the polynomial features

![poli](/pictures/select_k_linreg_poli20.png)

  The graph shows the performance of the training and validation dataset depending on the degree of polynomials. As you can see, there is   a relevant overfitting by the 3th and 4th-degree polynomial regression. The 2nd degree polynomial regression performs much better than     the original linear regression  (degree=1st) and shows no overfitting.

  3. I tested the Ridge and the Lasso regression on the dataset: Both regressions have a regularization term, which try to prevent overfitting. The key difference between the two regressions is how they assign the penalty to the coefficients:
    * The Ridge regression is trained with L2 prior as regularizer: adds penalty equivalent to the square of the magnitude of coefficients
    * The Lasso regression is trained with L1 prior as regularizer: adds penalty equivalent to absolute value of the magnitude of coefficients
  
  I used GridSearchCV to tune the regularization parameter (alpha): I tried different values between 10e-7 and 10e5, but for the             parameters smaller 1000 there was no relevant difference in the model performance. The R2 score of both Lasso and Ridge regression was     very similar to the basic linear regression model.

  The following graphic shows the R2 score of the Ridge regression:

![Ridge](/pictures/ridge.png)

#### 2. Decision trees:

I tried several models to find the optimal parameters for the decision tree (by means of GridSearchCV). The optimal decision tree model was the following:
* max_depth=10
* max_features='auto’
* min_samples_leaf=30
* min_samples_split=50

The performance of the model was  similar both on the training and on the validation set: 92.7% and 91.6% respectively. The two most important features were the age and the performance of the car. Below is a graph of the first 3 layers.

![Dtree](/pictures/Decision_tree_depth_3.png)

#### 3. Random forest:
The random forest regressor fits a number of decision trees on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. The parameters of the optimal decision tree model (last paragraph) were used. The number of estimators were set to 100, that could improve the performance of the model by 1% and also reduce the mean square error. (R2 score on the training sample ist 93.6 % and on the validation sample is 92.8%.)

#### 4. Support vector machines:
I tried to fit an SVR model with different kernels. The ‘rbf’ kernel achieved the highest R2 score with 88.5%. Compared to the other models, the support vector machine had a lower performance both on the training and on the validation set.


#### 5. AdaBoost Regressor:
I also fitted an AdaBoost Regressor with the optimal decision tree. This regressor had the best performance with 95.8% on the training set (and 93.8% on the validation set). I also checked the performance with different number of input variables as the following graph shows.

![Adaboost](/pictures/adaboost.png)

If I used 9 variables in the AdaBoost regression, the performance of the fitted model decreased only by 0.1%. 

## Use of the model

I used the flask library for creating an interface. After filling out the relevant features, the model calculates the expected price of the car. For starting the interface type 

`python app.py `

on the command line. The following interface will appear:

![App](/pictures/Interface.png)

By filling out all the features and clicking on the Calculate button, the expected price of the car appears on the top of the screen.

## Refinements of the model:
* I could have aggregated different models to get the final price (voting) 
* Involving more features when scraping the data (e.g. description could be analysed with NLP)
