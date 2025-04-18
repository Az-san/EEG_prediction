import os
import time
import threading
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import tensorflow as tf
from scipy import signal
import asyncio
import websockets
from preprocess_eeg import *

#import EEGmodels as eg
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
filerate = 0.001
writerate = 1
channelsdict = {}
timecount = 0

dat_lock = threading.Lock()
if exists("testoutpd.tsv"):
    os.remove("testoutpd.tsv")

class eeg_realtime():
    def __init__(self, labels, model, timeslice, relevant_channels, prediction_interval):
        self.label_names = labels
        self.preprocessor = preprocssor(labels)
        self.socket_thread = threading.Thread(target=self.runasyn) # socket connection runner
        self.model = model
        #self.model2 = model2
        self.timeslice = timeslice
        self.relevant_channels = relevant_channels
        self.currpred = None
        self.time = prediction_interval

        live_plotter_thread = threading.Thread(target=self.plot_live, args=())
        live_plotter_thread.start()
        
    def plot_live(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.main_window = MainWindow()
        self.main_window.show()
        sys.exit(self.app.exec_())

    def filestring(self):
        print('filestring')

    #preds_lock = threading.Lock()

    #socket connection
    # async def command(self):
    #     #async with websockets.connect('ws://127.0.0.1:5678') as websocket:
    #     #async with websockets.connect('ws://172.17.6.210:52350') as websocket:
    #     #async with websockets.connect('ws://172.17.6.135:5678') as websocket:
        
    #     while True:
            
    #         if self.currpred!=None:
    #             #async with websockets.connect('ws://172.17.6.210:52357') as websocket:
    #             async with websockets.connect('ws://127.0.0.1:5678') as websocket:
    #                 await websocket.send(str(self.currpred))
    #                 print(self.currpred,'sent.')
    #                 self.currpred=None
    #         else:
    #             await asyncio.sleep(1)
    #             print('sleep.')

    async def command(self):
        async with websockets.connect('ws://127.0.0.1:5678') as websocket:
        #async with websockets.connect('ws://172.17.6.210:52350') as websocket:
        #async with websockets.connect('ws://172.17.6.135:5678') as websocket:
        
            while True:
                
                if self.currpred!=None:
                    #async with websockets.connect('ws://172.17.6.210:52357') as websocket:
                    #async with websockets.connect('ws://127.0.0.1:5678') as websocket:
                    await websocket.send(str(self.currpred))
                    print(self.currpred,'sent.')
                    self.currpred=None
                else:
                    await asyncio.sleep(1)
                    print('sleep.')

    def runasyn(self):
        asyncio.run(self.command())
        #Read
    

    
    def make_prediction(self, wave, model):
        print('realtimepredictionwave',type(wave),wave.shape)
        if len(wave)>self.timeslice:
            wave = wave[:self.timeslice]
        currspect = self.preprocessor.convert_to_spectrogram(wave)
        print('prediction spectrogramshape',currspect.shape)
        #self.preprocessor.display_spectrogram(wave,currspect,'prediction')
        return np.argmax(model.predict(np.array([currspect]))[0])
        #return int(np.argmax(self.model.predict(tf.expand_dims([currspect], -1)), axis=1)[0])
        
    #overlap = 0.75 #4 slices per second
    #overlap= 0.67 # 3 slices per second
    #overlap = 0.80 # 5 slices per second
    #overlap = 0 # no overlap
    def get_num_lines(self,fname):
        with open(fname) as f:
            for i, _ in enumerate(f):
                pass
        return i + 1

    def read_file(self, filename,relevant_channels):
        if exists(filename) and os.stat(filename).st_size > 0:
            with open(filename, "r") as f:
                curr_end = self.get_num_lines(filename)
                #print(curr_end,curr_end-self.timeslice*8, curr_end-curr_end-self.timeslice*8)
                eeg_data = pd.read_csv(filename, encoding="Shift-JIS",sep='\t',skiprows=lambda x: x>0 and x<curr_end-self.timeslice*8 or x%8 not in relevant_channels and x!=0, usecols=['Value']).to_numpy().flatten()#[self.get_num_lines(filename)-8*self.timeslice-1:]
                #eeg_data = pd.read_csv(filename, encoding="Shift-JIS",sep='\t',skiprows=lambda x:x%8 not in self.relevant_channels and x!=0 and x>self.get_num_lines(filename)-8*self.timeslice, usecols=['Value'])
                time_data = pd.read_csv(filename, encoding="Shift-JIS",sep='\t',skiprows=lambda x: x>0 and x<curr_end-self.timeslice*8 or x%8 not in relevant_channels and x!=0, usecols=['Time']).to_numpy().flatten()#[self.get_num_lines(filename)-8*self.timeslice-1:]
                
            return eeg_data, time_data#.to_numpy().flatten()[self.get_num_lines(filename)-8*self.timeslice-1:]
                    

    def real_time(self, filename):
        self.socket_thread.start()
        #print('stated realtime')
        while True:
            currwave, currtime = self.read_file(filename, self.relevant_channels) # 2 channel averages
            #currwave2, currtime2 = self.read_file(filename, [2])
            #print(currtime, currtime[-1])
            self.currtime = currtime[-1]
            #make prediction
            self.currpred = self.make_prediction(currwave, self.model)
            # self.currpred2 = self.make_prediction(currwave2, self.model2)
            # self.currpred = int((self.currpred1+self.currpred2)/2)
            print('HERE',self.currpred)
            self.main_window.set_plot_pred(self.currpred, self.currtime)
            print(self.currpred)
            #send through socket

            #wait
            time.sleep(self.time/1000)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.currpred = None
        # ay = self.graph_show.getAxis('left')
        # ticks = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        # ay.setTicks([[(v, str(v)) for v in ticks ]])

        self.x = []
        self.y = []
        self.data_line = self.graphWidget.plot(self.x, self.y)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.updateplot)
        self.timer.start()
        self.time_count = 0
    
    def updateplot(self):
        x_view_range = 5
        plot_time_interval = 1
        if self.currpred!=None:
            #print(self.currpred,self.currtime)
            self.x.append(self.currtime-0.98)
            self.x.append(self.currtime)
            self.y.append(self.currpred)
            self.y.append(self.currpred)
            self.time_count+=1
            if len(self.x)>x_view_range:
                self.graphWidget.setXRange(self.currtime-x_view_range-1,self.currtime+5,padding=0)
            self.graphWidget.setYRange(-1,2,padding=0)
            self.data_line.setData(self.x, self.y)
            self.currpred = None

    def set_plot_pred(self, pred, time):
        self.currpred = pred
        self.currtime = time
