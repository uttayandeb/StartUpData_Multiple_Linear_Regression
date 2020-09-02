#Prepare a prediction model for profit of 50_startups data.


############################ Multilinear Regression #################

#importing required packages 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import csv

# loading the data
Startup_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Multiple_linear_regression\\50_Startups.csv", encoding= 'unicode_escape')
 
#csv.reader(open('C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Multiple_linear_regression\\Toyota_Corolla.csv', newline='', encoding='utf-8'))



######################### Data Cleaning #############################








# to get top 6 rows
Startup_Data.head(6) #  

Startup_Data.shape  #(50, 5)
Startup_Data.dtypes#  column "State" is of type object so we have to create dummy variable

 #number of null values

Startup_Data.info()# so there are no null values in the data
 
Startup_Data.columns# Index(['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'], dtype='object')

# number of unique values 
Startup_Data.nunique()
 
#replacing the object type into boolean of the column State ie creating dummy variables

Startup_Data.replace(to_replace=['New York', 'California','Florida'],value= ['0', '1','2'], inplace=True)


Startup_Data.State

 
################## EDA(Exploratory Data analysis) ##################

#1st moment business decision
Startup_Data.mean()

Startup_Data.median()

Startup_Data.mode()
 

#2nd moment busines decision
Startup_Data.var()  
                 
Startup_Data.std()            

 

# 3rd and 4th moment business decision
Startup_Data.skew()

Startup_Data.kurt()
 
#### Graphical representation   #########
                  
plt.hist(Startup_Data.Profit)
 
plt.boxplot(Startup_Data.Profit)#we have an outlier at lower extream
 

plt.hist(Startup_Data['R&D Spend'])
 
plt.boxplot(Startup_Data['R&D Spend'])# no outliers

 
## llly  we can check the boxplot for rest of the columns


plt.plot(Startup_Data.Profit,Startup_Data['R&D Spend'],"bo");plt.xlabel("Profit");plt.ylabel("R&D Spend")
 
#so the profit is linearly depends on the R&D spend

plt.plot(Startup_Data.Profit,Startup_Data.Administration,"bo");plt.xlabel("Profit");plt.ylabel(" Administration")
 
#does  not depend linerly since the data is segregated 

plt.plot(Startup_Data.Profit,Startup_Data['Marketing Spend'],"bo");plt.xlabel("Profit");plt.ylabel(" marketing spend")
 

plt.plot(Startup_Data.Profit,Startup_Data.State,"bo");plt.xlabel("Profit");plt.ylabel("State")
 
#Florida is giving more profit as compare to other states














##Correlation between each columns


Startup_Data.Profit.corr(Startup_Data.Administration) # 0.20071656826872136 # correlation value between X and Y
# Since the correlation value is too low so there is no correlation between Profit and the money spend in Administration 


### or ### table format
Startup_Data.corr() 
#                 R&D Spend  Administration  Marketing Spend    Profit
#R&D Spend         1.000000        0.241955         0.724248  0.972900
#Administration    0.241955        1.000000        -0.032154  0.200717
#Marketing Spend   0.724248       -0.032154         1.000000  0.747766
#Profit            0.972900        0.200717         0.747766  1.000000

#or using numpy
np.corrcoef(Startup_Data.Profit,Startup_Data['Marketing Spend'])#array([[1.        , 0.74776572],

import seaborn as sns
sns.pairplot(Startup_Data)
 
# Correlation matrix 
correlation=Startup_Data.corr()
 


                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

         
############ Preparing MLR model  ####################
                
model1 = smf.ols('Profit~ Startup_Data.iloc[:,0]+Administration + Startup_Data.iloc[:,2] + State',data=Startup_Data).fit() # regression model

# Getting coefficients of variables               
model1.params 

# Summary
model1.summary()# Adj. R-squared:                  0.945




#### preparing different models based on each column

# preparing model based only on R&D Spend
model_s=smf.ols('Profit~ Startup_Data.iloc[:,0]',data = Startup_Data).fit()  
model_s.summary() #   Adj. R-squared:                  0.945


# Preparing model based only on Administration
model_a=smf.ols('Profit~Administration',data = Startup_Data).fit()  
model_a.summary() #  Adj. R-squared:                  0.020

# Preparing model based only on Marketing Spend
model_m=smf.ols('Profit~Startup_Data.iloc[:,2]',data = Startup_Data).fit()  
model_m.summary() # Adj. R-squared:                  0.550

# Preparing model based only on State
model_State=smf.ols('Profit~State',data = Startup_Data).fit()  
model_State.summary() # Adj. R-squared:                 -0.018






# Preparing model based only on R&D Spend,Administration and Marketing Spend
model_RAM=smf.ols('Profit~ Startup_Data.iloc[:,0]+Administration + Startup_Data.iloc[:,2]',data =Startup_Data).fit()  
model_RAM.summary() # Adj. R-squared:                  0.948



# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 45,46,48,49 is showing high influence so we can exclude that entire row
Startup_Data_new=Startup_Data.drop(Startup_Data.index[[45,46,48,49]],axis=0)


# Studentized Residuals = Residual/standard deviation of residuals





# X => A B C D 
# X.drop(["A","B"],axis=1) # Dropping columns 
# X.drop(X.index[[5,9,19]],axis=0)

#X.drop(["X1","X2"],aixs=1)
#X.drop(X.index[[0,2,3]],axis=0)


# Preparing model                  
model1_new = smf.ols('Profit~ Startup_Data_new.iloc[:,0]+Administration + Startup_Data_new.iloc[:,2] + State',data=Startup_Data_new).fit()   

# Getting coefficients of variables        
model1_new.params

# Summary
model1_new.summary() #    R-squared:                       0.963,little bit increased
# Confidence values 99%
print(model1_new.conf_int(0.01)) # 99% confidence level


# Predicted values of price
profit_pred = model1_new.predict(Startup_Data_new[['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']])
profit_pred

Startup_Data_new.head()


# calculating VIF's values of independent variables
rsq_R = smf.ols('Startup_Data_new.iloc[:,0]~ Administration+ Startup_Data_new.iloc[:,2] + State',data=Startup_Data_new).fit().rsquared  
vif_R = 1/(1-rsq_R )
print(vif_R ) #2.5489151853096614


rsq_A = smf.ols('Administration~ Startup_Data_new.iloc[:,0]+Startup_Data_new.iloc[:,2] + State',data=Startup_Data_new).fit().rsquared  
vif_A = 1/(1-rsq_A )
print(vif_A) #1.2373858776930344



rsq_M = smf.ols('Startup_Data_new.iloc[:,2]~ Startup_Data_new.iloc[:,0]+Administration + State',data=Startup_Data_new).fit().rsquared  
vif_M = 1/(1-rsq_M )
print(vif_M) #2.539141249531141


import preprocessing

Startup_Data_new['State'] = pd.Categorical(Startup_Data_new['State'])
Startup_Data_new['State'] = Startup_Data_new['State'].cat.codes
min_max_scaler = preprocessing.MinMaxScaler()
Startup_Data_new['State']= min_max_scaler.fit_transform(Startup_Data_new[['State']])





rsq_S = smf.ols(' State~ Startup_Data_new.iloc[:,0]+Administration + Startup_Data_new.iloc[:,2]',data=Startup_Data_new).fit().rsquared  
vif_S = 1/(1-rsq_S )
print(vif_S) #1.0050829293773804









##################### Storing vif values in a data frame ###################

d1 = {'Variables':['R&D Spend', 'Administration', 'Marketing Spend', 'State'],'VIF':[vif_R,vif_A,vif_M,vif_S]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As there is no  higher VIF value

# Added varible plot 
sm.graphics.plot_partregress_grid(model1_new)

# added varible plot for weight is not showing any significance 

######################## final model ###################################
final_model= smf.ols('Profit~ Startup_Data_new.iloc[:,0]+Administration + Startup_Data_new.iloc[:,2] + State',data=Startup_Data_new).fit()
final_model.params
final_model.summary() #Adj. R-squared:                  0.959
# As we can see that r-squared value has increased

profit_pred = final_model.predict(Startup_Data_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_model)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Startup_Data_new.Profit,profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(profit_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_model.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(profit_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
Startup_Data_train,Startup_Data_test  = train_test_split(Startup_Data_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols('Profit~Startup_Data_train.iloc[:,0]+Administration + Startup_Data_train.iloc[:,2] + State',data=Startup_Data_train).fit()
model_train1 = smf.ols('Profit~Startup_Data_test.iloc[:,0]+Administration + Startup_Data_test.iloc[:,2] + State',data=Startup_Data_test).fit()

# train_data prediction
train_pred = model_train.predict(Startup_Data_train)

# train residual values 
train_resid  = train_pred - Startup_Data_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(Startup_Data_test)

# test residual values 
test_resid  = test_pred - Startup_Data_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
