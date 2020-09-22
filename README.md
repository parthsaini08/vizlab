# vizlab
Library for visualizing machine learning models and for Data Preprocessing
# Use
This library is made to make visualizing easy for all the basic machine learning models including Regression , Classification and Clustering.
## Regression
The code for regression looks like : 

***from vizlab import visualize***  
***from sklearn.linear_model import LinearRegression***  
***import numpy as np***    
***X= 2 *np.random.randn(100,1)***    
***y=4+3*X+np.random.randn(100,1)***    
***visualize.viz_reg(LinearRegression(),X,y)***    

  ![Regression](/images/Regression.png)  
  
  
You can also do the same for more than one model i.e. by directly passing the list of all models you want to see the regression curve for.
Example : 

***visualize.viz_reg([LinearRegression(),Ridge()],X,y)*** 

## Plotting the Learning Curves for a regression model
The code for this looks like:

***import numpy as np*** 
***from vizlab import visualize***  
***from sklearn.linear_model import LinearRegression*** 
***X= 2 *np.random.randn(100,1)***  
***y=4+3*X+np.random.randn(100,1)***  
***visualize.plot_learning_curves_reg(LinearRegression(),X,y)***  

  ![Learning curves](/images/learning_curves.png)
## Classification 
The code for classification looks like :

***from vizlab import visualize*** 
***from sklearn.svm import SVC***  
***from sklearn.datasets import make_moons***  
***X,y=make_moons(n_samples=200)***  
***visualize.viz_clf(SVC(),X,y)***  

You can also do the same for more than one model i.e. by directly passing the list of all models you want to see the classification curve for.
Example :

***visualize.viz_clf([DecisionTreeClassifier(),SVC()],X,y)***  

## Clustering
The code for clustering looks like:

***from vizlab import visualize***  
***from sklearn.cluster import KMeans***  
***from sklearn.datasets import make_blobs***  
***X,y=make_blobs(n_samples=200)***  
***visualize.viz_cluster(KMeans(n_clusters=3),X)***  

Moreover if you want to cluster analysis i.e. if you want to know the number of clusters set the __help_clusters=True__ . You can also set the upper range and lower range for the number of clusters u want to examine the code for and accordingly set the values of upper_range and lower_range parameters. The analysis will be done on the basis of sillhoute score of the clusters in different cases.
Example:

***visualize.viz_cluster(KMeans(n_clusters=3),X,help_clusters=True,upper_range=6)***  

## Data Analysis
You can also data analysis on a given dataframe using this library . The data will be analysed and will be saved in the form of a pdf in your current working directory .
For this the code goes as follows : 

***import pandas as pd***  
***from vizlab import repo***  
***df=pd.read_csv('insurance.csv')***  
***repo.report(df)***  

Hence a pdf will be created which gives a basic overview of your data.








 
