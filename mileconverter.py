import pandas as pd
def main():
    i=0
    df=pd.read_csv("m_times.csv", usecols=["event_code","mark"])
    multiplier=(1609/1600)
    mile_filter=df["event_code"].str.contains("Mile",na=False)
    mile_rows=df[mile_filter]
    mile_rows["mark"] = mile_rows["mark"].astype(int)
    mile_rows["mark"]*=multiplier
    length=len(df["event_code"])
    for index,row in mile_rows.iterrows():
        print(f"Row {index}: {row}")
   
if __name__ == '__main__':  

    main()