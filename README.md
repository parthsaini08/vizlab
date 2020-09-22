# vizlab
Library for visualizing machine learning models and for Data Preprocessing
# Use
This library is made to make visualizing easy for all the basic machine learning models including Regression , Classification and Clustering.
## Regression
The code for regression looks like : 

***from vizlab import visualize***  
from sklearn.linear_model import LinearRegression  
import numpy as np  
X= 2 *np.random.randn(100,1)  
y=4+3*X+np.random.randn(100,1)  
visualize.viz_reg(LinearRegression(),X,y)  

You can also do the same for more than one model i.e. by directly passing the list of all models you want to see the regression curve for.
Example : 

visualize.viz_reg([LinearRegression(),Ridge()],X,y)
## Classification 
The code for classification looks like :
 
