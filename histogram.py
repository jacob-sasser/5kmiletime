
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main():
    data = pd.read_csv("sampledata2.csv", usecols=['event_code', 'minutes'])

    plt.hist([data[data.columns[0]], data[data.columns[1]]], label=[data.columns[0], data.columns[1]])
    print(data.columns)
    
    plt.xlabel('')
    plt.ylabel('Frequency')
    #plt.legend()
    plt.show()
if __name__ == '__main__':
    main()