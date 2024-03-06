import numpy as np
import pandas as pd
import re
import openpyxl
def filter():
    df=pd.read_csv("m_output.csv")
    has_5000m=df[(df['event_code']=='5000m')]
    
    common_athlete=set(df['athlete_id']).intersection(set(has_5000m['athlete_id']))
    df_filtered=df[df['athlete_id'].isin(common_athlete)]
    df_filtered.to_csv("m_test.csv")
    print(df_filtered)
    '''df=pd.read_csv("f_output.csv")
    df=df[(df['event_code']=='5000m')]
    df.to_csv("f_5000m.csv")'''

    print()
    
def main():
    
    dates_col = pd.read_csv("f_output.csv", usecols=["meet_days"])

    # Define pattern1 for two-day events that start at the end of a month
    pattern1 = r'(\w{3}) (\d{2}) - (\w{3}) (\d{2}) (\d{4})'
    
    # Define pattern2 for two-day events in the same month
    pattern2 = r'(\w{3}) (\d{2}) - (\d{2}) (\d{4})'

    pattern3 = r'(\w{3}) (\d{2})-(\d{2}) (\d{4})'
   
    dates_col['meet_days'] = dates_col['meet_days'].str.replace(pattern2, r'\1 \2', regex=True)
    dates_col['meet_days'] = dates_col['meet_days'].str.replace(pattern1, r'\1 \2 \4', regex=True)
    dates_col['meet_days'] = dates_col['meet_days'].str.replace(pattern3, r'\1 \2 \4', regex=True)
    dates_col.to_csv("w_times2.csv", index=False)

    #repeat for men times
    dates_col_M = pd.read_csv("m_output.csv", usecols=["meet_days"])

    # Define pattern1 for two-day events that start at the end of a month
    pattern1 = r'(\w{3}) (\d{2}) - (\w{3}) (\d{2}) (\d{4})'
    
    # Define pattern2 for two-day events in the same month
    pattern2 = r'(\w{3}) (\d{2}) - (\d{2}) (\d{4})'
    
    pattern3 = r'(\w{3}) (\d{2})-(\d{2}) (\d{4})'
   
    dates_col_M['meet_days'] = dates_col_M['meet_days'].str.replace(pattern2, r'\1 \2', regex=True)
    dates_col_M['meet_days'] = dates_col_M['meet_days'].str.replace(pattern1, r'\1 \2 \5', regex=True)
    dates_col_M['meet_days'] = dates_col_M['meet_days'].str.replace(pattern3, r'\1 \2 \4', regex=True)
    dates_col_M.to_csv("M_times2.csv", index=False)

   


    print(dates_col_M)

if __name__ == '__main__':
    filter() 
