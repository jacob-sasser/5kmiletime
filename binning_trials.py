# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Imports and Data Allocation
# 

# %%
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import statsmodels as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import seaborn as sns

'''
File columns needed:
minutes in double minutes
day and month
meet_start date
class of and year

'''


# %%
# Import Male Data
meets_df = pd.read_csv("m_times", error_bad_lines=False)

# Cleaning Data to only use 5000m and use> 12 min meets
meets = meets_df[meets_df['event_code'] == '5000m']
meets = meets[meets['minutes'] > 12]


# %%
# number of days from start of season, from ath_progression.ipynb
def date_to_days(date:list) -> int:
    first_day = [1,8]
    month_days = {8:31,9:30,10:31,11:30,12:31}

    if date[1] < first_day[1]: return None

    days = date[0] - first_day[0]

    while date[1] > first_day[1]:
        days += month_days[date[1]-1]
        date = [date[0],date[1]-1]

    return days


# %%
# add day in season to the records
meets['day_in_season'] = meets.apply(lambda x: date_to_days([x['day'],x['month']]), axis=1)
meets.head()

# %% [markdown]
# ## Binning Start
# %% [markdown]
# Clean Dataset

# %%
def clean_dataset(df):
    '''
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
    '''
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# %% [markdown]
# Date Time

# %%
meets['meet_date'] = pd.to_datetime(meets['meet_start'], format='%b %d %Y')
meets['day_of_week'] = meets['meet_date'].dt.dayofweek

meets['grade'] = meets.apply(lambda x: 13 + x['year'] - x['class_of'], axis=1)

# %% [markdown]
# Find Season Best

# %%
def bestTimes(x):
  temp = meets[(meets['athlete_id'] == x['athlete_id']) & (meets['year'] == x['year'])]['minutes'].to_list() 
  if (temp):
    return min(temp)
  return None

meets['s_best'] = meets.apply(lambda x: bestTimes(x), axis=1)
meets

# %% [markdown]
# Bin Athelete Id By the minimum time

# %%
m_bins = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,36]
meets['s_best_bin'] = pd.cut(meets['s_best'], m_bins)

print(meets.head(20))

# %% [markdown]
# For loop for displaying results

# %%
def r2(X, y, model):
  # compute with formulas from the theory
  model.fit(X, y)
  yhat = model.predict(X)
  SS_Residual = sum((y-yhat)**2)       
  SS_Total = sum((y-np.mean(y))**2)     
  r_squared = 1 - (float(SS_Residual))/SS_Total
  # adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
  print('Rsquared from the function', r_squared)


# %%
for _bin in meets['s_best_bin'].unique():
    if _bin is not np.nan:
          temp = meets[(meets['s_best_bin'] == _bin)]
          meets_y = temp[['minutes']]
          meets_X = temp[['day_in_season']]
          meets_X_train = meets_X[:-20]
          meets_X_test = meets_X[-20:]

          meets_y_train = meets_y[:-20]
          meets_y_test = meets_y[-20:]

          regr = linear_model.LinearRegression()
          regr.fit(clean_dataset(meets_X_train), clean_dataset(meets_y_train))

          
          # Make predictions using the testing set
          meets_y_pred = regr.predict(clean_dataset(meets_X_test))
          
          # The coefficients
          print('Intercept: ', regr.intercept_)

          print('Coefficients: \n', regr.coef_)

          print('Mean squared error: %.2f'
                % mean_squared_error(meets_y_test, meets_y_pred))

          print('Coefficient of Determination: %f'
                % r2_score(meets_y_test, meets_y_pred))
          
          #r2(meets_X, meets_y, regr)
          
          
          import statsmodels.formula.api as sm
          result = sm.ols(formula="minutes ~ day_in_season", data=temp).fit()
          print(result.summary())
          print(result.rsquared)

          # Plot outputs
          plt.scatter(meets_X, meets_y,  color='black')
          plt.scatter(meets_X_test, meets_y_test,  color='red')
          plt.plot(meets_X_test, meets_y_pred, color='blue', linewidth=3)

          plt.title(_bin)
          #plt.legend()
          plt.show()

# %% [markdown]
# Year to year improvement (WIP)

# %%
def yearImprovment(x):
  temp = meets[(meets['athlete_id'] == x['athlete_id']) & (meets['year'] == x['year'])]['minutes'].to_list() 
  if (temp):
    return min(temp)
  return None


# %%
'''
a = 0;
for _bin in meets['athlete_id'].unique():
  if _bin is not np.nan:
    print(meets[_bin]['s_best_bin'])
    a+=1
'''

# %% [markdown]
# ##Find Athelete Progression grade by grade
# 
# %% [markdown]
# ### Freshman testing

# %%
sophomore = meets[(meets['grade']==10.0)]
target = []
for _bin in meets['s_best_bin'].unique():
    if _bin is not np.nan:
      freshman = meets[(meets['s_best_bin'] == _bin) & (meets['grade']==9.0)]
      #sophomore = meets[(meets['s_best_bin'] == _bin) & (meets['grade']==10.0)]
      
      new_sophomore = sophomore['athlete_id'].isin(freshman['athlete_id'])
      #print(sophomore[new_sophomore].s_best)

      #print(sophomore.loc[sophomore['athlete_id']==7279456].s_best)



      #print(freshman.loc[meets['grade'] == 9.0].athlete_id)
      #print(findSecondary(freshman, '9353519'))
      #print(freshman.loc[meets['athlete_id'] == 9353519].s_best)
      #print(freshman.loc[(meets['grade'] == 9.0) & (meets['grade'] == 10.0)].athlete_id)

      #freshman.set_index("athlete_id", inplace=True)
      #sophomore.set_index("athlete_id", inplace=True)
      rows = []
      temp['delta'] = 0;
      for athID in freshman['athlete_id']:
        if not sophomore.loc[meets['athlete_id'] == athID].empty:
          #print(athID)
          #print("First: ", sophomore.loc[meets['athlete_id'] == athID].s_best.tolist()[0], "Second: ", freshman.loc[meets['athlete_id'] == athID].s_best.tolist()[0] )
          #print(sophomore.loc[meets['athlete_id'] == athID].s_best.tolist()[0] - freshman.loc[meets['athlete_id'] == athID].s_best.tolist()[0])
        #print(freshman.loc[meets['athlete_id'] == athID].s_best.tolist()[0] - sophomore.loc[meets['athlete_id'] == athID].s_best.tolist()[0])
      
          #ax = temp.plot.kde()
          rows.append([athID, sophomore.loc[meets['athlete_id'] == athID].s_best.tolist()[0] - freshman.loc[meets['athlete_id'] == athID].s_best.tolist()[0] ])

      #print(rows)
      temp = pd.DataFrame(rows, columns=["athID", "delta"])
      temp.drop_duplicates(subset ="athID",
                     keep = 'first', inplace = True)
      print(_bin)
      print(temp.head())
      
      
      print("Mean: ", temp['delta'].mean())
      target.append([_bin, temp['delta'].mean(), temp['delta'].std() ])
      print("Descriptive Statistics: ")
      print(temp['delta'].describe())


      plt.hist(temp['delta'], density=True, bins=20)  # density=False would make counts
      sns.distplot(temp['delta'], hist=False)
      plt.ylabel('Probability')
      plt.xlabel('Data');
      plt.show()
year_target =  pd.DataFrame(target, columns=["bin", 'delta_mean', "std"])
year_target['grade']= 9
plt.hist(year_target["delta_mean"], density=True, bins=20)
plt.show()

# %% [markdown]
# Sophomore Testing

# %%
sophomore = []
junior = meets[(meets['grade']==11.0)]
target = []
for _bin in meets['s_best_bin'].unique():
    if _bin is not np.nan:
      sophomore = meets[(meets['s_best_bin'] == _bin) & (meets['grade']==10.0)]
      
      rows = []
      temp['delta'] = 0;
      for athID in sophomore['athlete_id']:
        if not junior.loc[meets['athlete_id'] == athID].empty:
          rows.append([athID, junior.loc[meets['athlete_id'] == athID].s_best.tolist()[0] - sophomore.loc[meets['athlete_id'] == athID].s_best.tolist()[0] ])

      temp = pd.DataFrame(rows, columns=["athID", "delta"])
      temp.drop_duplicates(subset ="athID",
                     keep = 'first', inplace = True)
      print(_bin)
      print(temp.head())
      
      
      print("Mean: ", temp['delta'].mean())
      target.append([_bin, temp['delta'].mean(), temp['delta'].std() ])
      print("Descriptive Statistics: ")
      print(temp['delta'].describe())


      plt.hist(temp['delta'], density=True, bins=20)  # density=False would make counts
      sns.distplot(temp['delta'], hist=False)
      plt.ylabel('Probability')
      plt.xlabel('Data');
      plt.show()
year_target2 =  pd.DataFrame(target, columns=["bin", 'delta_mean', "std"])
year_target2['grade']= 10
plt.hist(year_target2["delta_mean"], density=True, bins=20)
plt.show()

year_target.append(year_target2)


# %%
junior = []
senior = meets[(meets['grade']==12.0)]
target = []
for _bin in meets['s_best_bin'].unique():
    if _bin is not np.nan:
      junior = meets[(meets['s_best_bin'] == _bin) & (meets['grade']==11.0)]
      
      rows = []
      temp['delta'] = 0;
      for athID in junior['athlete_id']:
        if not senior.loc[meets['athlete_id'] == athID].empty:
          rows.append([athID, senior.loc[meets['athlete_id'] == athID].s_best.tolist()[0] - junior.loc[meets['athlete_id'] == athID].s_best.tolist()[0] ])

      temp = pd.DataFrame(rows, columns=["athID", "delta"])
      temp.drop_duplicates(subset ="athID",
                     keep = 'first', inplace = True)
      print(_bin)
      print(temp.head())
      
      
      print("Mean: ", temp['delta'].mean())
      target.append([_bin, temp['delta'].mean(), temp['delta'].std() ])
      print("Descriptive Statistics: ")
      print(temp['delta'].describe())


      plt.hist(temp['delta'], density=True, bins=20)  # density=False would make counts
      sns.distplot(temp['delta'], hist=False)
      plt.ylabel('Probability')
      plt.xlabel('Data');
      plt.show()
year_target3 =  pd.DataFrame(target, columns=["bin", 'delta_mean', "std"])
year_target3['grade']= 11
plt.hist(year_target3["delta_mean"], density=True, bins=20)
plt.show()

year_target.append(year_target3)


# %%
year_target = year_target[year_target['grade'] < 10]
print(year_target.head())


next_year = meets[meets['year'] == 2019]
next_year = next_year[['athlete_id', 'team_id', 'class_of', 's_best', 's_best_bin', 'grade']]
next_year = next_year[next_year['grade'] < 12]


for i in next_year.index:
  if next_year.loc[i,'grade'] == 9:
    for j in year_target.index:
      if next_year.loc[i, 's_best_bin'] == year_target.loc[j, 'bin']:
        next_year.loc[i, 'delta_mean'] = year_target.loc[j, 'delta_mean']
        next_year.loc[i, 'std'] = year_target.loc[j, 'std']
  if next_year.loc[i,'grade'] == 10:
    for j in year_target2.index:
      if next_year.loc[i, 's_best_bin'] == year_target2.loc[j, 'bin']:
        next_year.loc[i, 'delta_mean'] = year_target2.loc[j, 'delta_mean']
        next_year.loc[i, 'std'] = year_target2.loc[j, 'std']
  if next_year.loc[i,'grade'] == 11:
    for j in year_target3.index:
      if next_year.loc[i, 's_best_bin'] == year_target3.loc[j, 'bin']:
        next_year.loc[i, 'delta_mean'] = year_target3.loc[j, 'delta_mean']
        next_year.loc[i, 'std'] = year_target3.loc[j, 'std']



#next_year.loc[i, 'delta_mean'] = year_target.loc[next_year.loc[i,'bin'], 'delta_mean'] 


# %%
next_year.drop_duplicates(subset ="athlete_id",
                     keep = 'first', inplace = True)

next_year


# %%
next_year.to_csv('bin_statistics.csv')
year_target.to_csv('freshman.csv')
year_target2.to_csv('sohphomore.csv')
year_target3.to_csv('junior.csv')

