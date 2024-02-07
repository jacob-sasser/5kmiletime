import numpy as np
import pandas as pd
import re
def data_filter(data, meetday):
    dates_col= pd.read_excel(data, usecols=meetday)
    #pattern 1 is two day events that start at the end of a month
    pattern1= r"(\w{3}) (\d{2}) - (\w{3}) (\d{2}) (\d{4})"
    #patter 2 is the two day events in the same month
    pattern2=r"(\w{3}) (\d{2}) - ({\d{2}) (\d{4})"

    pattern1match=re.match(pattern1, dates_col)
    
    
def main():
    data_filter("w_times.xlsx","meet_days")

if __name__ == '__main__':
    main()