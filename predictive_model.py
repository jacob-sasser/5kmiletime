import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as sfm

def main():
    df = pd.read_csv("m_times.csv")
    sm.datasets.get_rdataset(df).data
    results=