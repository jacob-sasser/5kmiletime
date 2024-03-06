import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as sfm
import os

def main():
    df = pd.read_csv("m_times.csv")

    mod=sm_data=sm.OLS(endog="mark",exog="event_code" )
    res=mod.fit()
    print(res.summary())

if __name__ =="__main__":
    main()
    