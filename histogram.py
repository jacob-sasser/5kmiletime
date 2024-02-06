
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main():
    data = pd.read_csv("sampledata2.csv", usecols=['event_code', 'minutes'])
    #event codes
    #plt.hist([data[data.columns[0]]], bins =10)

    #minutes
    plt.hist([data[data.columns[1]]], label=[data.columns[1]], bins=20)
    
    
    plt.show()
if __name__ == '__main__':
    main()