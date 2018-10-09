# Major Leagues - Football (Soccer)

It is a Data Regression Project

# What it Does?
  - It predicts the score of a team based on the other metrics.
  
# Datset Used
- Soccer SPI : https://github.com/fivethirtyeight/data/tree/master/soccer-spi
- ###### It contains 3 Data Files.


        SPI Global Rankings - Ranking of all soccer teams (club)
        SPI Global Rankings - Ranking of all International soccer teams.
        SPI all Matches - Metrics of all the matches
- We will be using **SPI all matches** dataset to predict the goals for each team

# Regression
- We will be applying two types of regression on data
-- **Linear Regression**
-- **Random Forest Regression**

# Data Preparation
- **NaN Values**
  - Some of the mathces didn't have **Scores**. Those records were removed (No way to compare the prediction)
  - Some of the other fields had NaN. Mean values of the respective column is used.


# Analysis
#### Linear Regression:
- Data has been split into Train and Test.
- Linear Regression has been applied for both the teams to predict the score.
- Below graphs shows the Residual plot for both the teams.

![Residual Plot 1](https://raw.githubusercontent.com/nareshkumar66675/SoccerSPI/master/reports/ResidualTeam1.png "Residual Plot 1")  |  ![Residual Plot 2](https://raw.githubusercontent.com/nareshkumar66675/SoccerSPI/master/reports/ResidualTeam2.png " Residual Plot 2")

#### Random Forest Regression:
- 100 Decision trees has been used for the prediction.
- Computed Decision Tree
  - [Team 1](https://raw.githubusercontent.com/nareshkumar66675/SoccerSPI/master/reports/RFTree1.png)
  - [Team 2](https://raw.githubusercontent.com/nareshkumar66675/SoccerSPI/master/reports/RFTree2.png)

#### Results:
- Random forest performed well in training data set. 
- Linear regression model gave almost same result in both Traning and Testing.
- Below graph shows the comparison of each model.

![Mean Square Error](https://raw.githubusercontent.com/nareshkumar66675/SoccerSPI/master/reports/MSE.png "MEan Square Error")


# Project Struture

##### Src
- SoccerSPI.py - python file exported from Jupyter
##### Notebooks
- SoccerSPI.ipynb - Jupyter notebook
##### Data
- External - SoccerSPI Data
##### Reports
- Trees - Decision Trees
- Graphs - Residual & Mean Square Error

***


  
