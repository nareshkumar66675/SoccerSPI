
# coding: utf-8

# In[1]:


from pandas import DataFrame, read_csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

# Enable inline plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read FootBall(Soccer - I'm not an American) Data 

matchFilePath = r'C:\Users\kumar\OneDrive\Documents\Projects\SoccerSPI\data\external\spi_matches.csv'
matchDF = pd.read_csv(matchFilePath)
matchDF.head(10)


# In[3]:


regDF = matchDF.drop(columns=['date','league_id','league','team1','team2'])
regDF.describe()


# In[4]:


regDF.dropna(subset=['score1','score2'],inplace=True)


# In[5]:


# Fill Empty or NaN values with the mean
regDF.fillna(regDF.mean(),inplace=True)
regDF.isna().sum()


# In[ ]:





# In[6]:


noScoreDF = regDF.drop(columns=['score1','score2'])


# In[7]:


# Split Total Data into Train and Test
from sklearn.model_selection import train_test_split

team1X_train, team1X_test, team1Y_train, team1Y_test = train_test_split(noScoreDF, regDF.score1, random_state=1)


# In[8]:


from sklearn.linear_model import LinearRegression

# Linear Regression 

team1Reg = LinearRegression()
team1Reg.fit(team1X_train,team1Y_train)


# In[9]:


team1Predict_train = team1Reg.predict(team1X_train)
team1Predict_test = team1Reg.predict(team1X_test)


# In[10]:


# Print Linear Regression Mean Square Error for Team 1

lfTrainTeam1MSE = round(np.mean(np.subtract(team1Y_train.values,team1Predict_train) ** 2),3)
print(lfTrainTeam1MSE)
lfTestTeam1MSE = round(np.mean(np.subtract(team1Y_test.values,team1Predict_test) ** 2),3)
print(lfTestTeam1MSE)


# In[11]:


plt.scatter(team1Predict_train,np.subtract(team1Predict_train,team1Y_train.values),c='b')
plt.scatter(team1Predict_test,np.subtract(team1Predict_test,team1Y_test.values),c='r')
plt.hlines(y=0,xmin=0,xmax=5)

plt.title('Residual Plot for Team 1 - Train:Blue and Test:Red')


# In[12]:


from sklearn.model_selection import train_test_split

team2X_train, team2X_test, team2Y_train, team2Y_test = train_test_split(noScoreDF, regDF.score2, random_state=5)

from sklearn.linear_model import LinearRegression
team2Reg = LinearRegression()
team2Reg.fit(team2X_train,team2Y_train)


# In[13]:


# Print Linear Regression Mean Square Error for Team 2

team2Predict_train = team2Reg.predict(team2X_train)
team2Predict_test = team2Reg.predict(team2X_test)

lfTrainTeam2MSE = round(np.mean(np.subtract(team2Y_train.values,team2Predict_train) ** 2),3)
lfTestTeam2MSE = round((np.mean(np.subtract(team2Y_test.values,team2Predict_test) ** 2)),3)

print(lfTrainTeam2MSE)
print(lfTestTeam2MSE)


# In[14]:


plt.scatter(team2Predict_train,np.subtract(team2Predict_train,team2Y_train.values),c='b')
plt.scatter(team2Predict_test,np.subtract(team2Predict_test,team2Y_test.values),c='r')
plt.hlines(y=0,xmin=0,xmax=5)

plt.title('Residual Plot for Team 2 - Train:Blue and Test:Red')


# In[15]:


# Regression Model - Team 1
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 100 decision trees
rfTeam1 = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfTeam1.fit(team1X_train,team1Y_train);


# In[16]:


# Print Random Forest Regression Mean Square Error for Team 1

rfPredictTrainTeam1 = rfTeam1.predict(team1X_train)
# Calculate the Square errors
rfErrorTrainTeam1 = (rfPredictTrainTeam1 - team1Y_train)**2
rfTrainTeam1MSE =  round(np.mean(rfErrorTrainTeam1),3)
print('Mean Square Error Train:', rfTrainTeam1MSE)



rfPredictTeam1 = rfTeam1.predict(team1X_test)
rfErrorTeam1 = (rfPredictTeam1 - team1Y_test)**2
rfTestTeam1MSE = round(np.mean(rfErrorTeam1),3)
print('Mean Square Error Test:', rfTestTeam1MSE)


# In[17]:


# Regression for Team 2

from sklearn.ensemble import RandomForestRegressor
rfTeam2 = RandomForestRegressor(n_estimators = 100, random_state = 62)
rfTeam2.fit(team2X_train,team2Y_train);


# In[18]:


# Print Random Forest Regression Mean Square Error for Team 2

rfPredictTrainTeam2 = rfTeam2.predict(team2X_train)
# Calculate the Square errors
rfErrorTrainTeam2 = (rfPredictTrainTeam2 - team2Y_train)**2
# Print out the mean absolute error (mae)
rfTrainTeam2MSE =  round(np.mean(rfErrorTrainTeam2),3)
print('Mean Square Error Train:', rfTrainTeam2MSE)


# Use the forest's predict method on the test data
rfPredictTeam2 = rfTeam2.predict(team2X_test)
# Calculate the absolute errors
rfErrorTeam2 = (rfPredictTeam2 - team2Y_test) **2
# Print out the mean absolute error (mae)
rfTestTeam2MSE = round(np.mean(rfErrorTeam2),3)
print('Mean Square Error Test:', rfTestTeam2MSE)


# In[19]:



# Visualize the Decision Tree
from sklearn.tree import export_graphviz
import pydot
tree = rfTeam1.estimators_[50]
export_graphviz(tree, out_file = r'C:\Users\kumar\OneDrive\Documents\Projects\SoccerSPI\reports\tree.dot', feature_names = list(noScoreDF.columns.values), rounded = True, precision = 1)


# In[20]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

(graph, ) = pydot.graph_from_dot_file(r'C:\Users\kumar\OneDrive\Documents\Projects\SoccerSPI\reports\tree.dot')
# Write graph to a png file
graph.write_png(r'C:\Users\kumar\OneDrive\Documents\Projects\SoccerSPI\reports\tree1.png')


# In[21]:


# Plot the Mean Square Error for Comparison
n_groups = 4
train_mse = (lfTrainTeam1MSE, lfTrainTeam2MSE,rfTrainTeam1MSE,rfTrainTeam2MSE)
test_mse = (lfTestTeam1MSE,lfTestTeam2MSE,rfTestTeam1MSE,rfTestTeam2MSE)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, train_mse, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train')
 
rects2 = plt.bar(index + bar_width, test_mse, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test')
 
plt.xlabel('Regression Method & Teams')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error of each Methods & Teams')
plt.xticks(index + bar_width, ('Linear - Team 1', 'Linear - Team 2', 'RF - Team 1', 'RF - Team 2'))
plt.legend()
 
plt.tight_layout()

