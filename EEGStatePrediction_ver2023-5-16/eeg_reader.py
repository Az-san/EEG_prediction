# reads the DAQMaster eeg file
import numpy as np
import pandas as pd
import os
import time




class read_file:
    def __init__(self, filename, relevant_channels):
        self.filename = filename
        self.relevant_channels = relevant_channels
    
    def read(self):
        print(self.filename)
        if os.path.exists(self.filename) and os.stat(self.filename).st_size > 0:
            eeg_data = pd.read_csv(self.filename, encoding="Shift-JIS",sep='\t',skiprows=lambda x:x%8 not in self.relevant_channels and x!=0, usecols=['Value'])
            return eeg_data.to_numpy().flatten()
        else:
            print('file not found')


