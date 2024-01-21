
import matplotlib.pyplot as pyplot
import numpy as np
import pandas
def main():
    data=pandas.read_csv("sampledata2.csv",usecols=['event_code','minutes'])
    data.hist()
if __name__ == '__main__':
    main()