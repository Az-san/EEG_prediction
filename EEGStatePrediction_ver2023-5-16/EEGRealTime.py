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

import EEGmodels as eg
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
filerate = 0.001
writerate = 1
channelsdict = {}
timecount = 0
currpred = None
dat_lock = threading.Lock()
if exists("testoutpd.tsv"):
    os.remove("testoutpd.tsv")

def filestring():
    print('filestring')

preds_lock = threading.Lock()
async def command():
    global currpred
    global preds_lock
    async with websockets.connect('ws://127.0.0.1:5678') as websocket:
        while True:

            if currpred != None:
                preds_lock.acquire()
                await websocket.send(str(currpred))
                #print(currpred[0])
                currpred = None
                preds_lock.release()
            else:
                await asyncio.sleep(0.2)

def runasyn():
    asyncio.run(command())
    #Read

t1 = threading.Thread(target=runasyn)



def continuous_write(fullfile):
    timedout = False
    timecount = 0
    stepsize = int((writerate/filerate)*8)
    #print(stepsize)
    #print('threeading'+str(arg1))
    #with open("testoutpd.tsv", "a+") as f:
    #    f.write(...)

    #f = open("testoutpd.tsv","a")


    #f.write('TypeSensor	Time	Ch	Value\n')
    #f.close()
    datcount = 0
    first = True
    finish = False
    while not finish:

        if(datcount+stepsize>len(fullfile)):
            stepsize = len(fullfile)-datcount
            finish = True
        partialfile = fullfile.iloc[datcount:datcount+stepsize].copy()

        #pfs = partialfile.to_string(index=False, header=0)
        #f.write(pfs+'\n')
        if first:
            with open("testoutpd.tsv", "a") as f:
                #partialfile.to_csv(f, sep='\t', encoding="Shift-JIS", mode='a', index=False)
                partialfile.to_csv(f, sep='\t', encoding="Shift-JIS", index=False)
            first = False
        else:
            with open("testoutpd.tsv", "a") as f:
                #partialfile.to_csv(f, sep='\t', encoding="Shift-JIS",mode='a',index=False,header=0)
                partialfile.to_csv(f, sep='\t', encoding="Shift-JIS",index=False,header=0)
        time.sleep(writerate)
        #print('filewrite',timecount, datcount)
        timecount+=writerate
        #if timecount>=1:
        #    break;
        datcount+=stepsize
    print('Ended write loop')



def simulate_write(filename):
    print('starting sim write file')
    with open(filename, "r") as f:
        fullfile = pd.read_csv(f, encoding="Shift-JIS", sep='\t')
    #print(fullfile)
    #print(fullfile['Time'][0])
    #print(fullfile[['Time','Ch']])
    #print(fullfile.iloc[:, 0:2])
    #print(fullfile.iloc[0, 0:2].copy())
    #print(fullfile.iloc[0].copy())
    #partialfile = fullfile.iloc[0:8].copy()
    #print(partialfile)
    #print(partialfile.keys())
    #partialfile.to_csv('testoutpd.tsv', sep='\t',encoding="Shift-JIS")
    #partialfile = fullfile.iloc[0:16].copy()
    #partialfile.to_csv('testoutpd.tsv', sep='\t', encoding="Shift-JIS")
    t = threading.Thread(target=continuous_write, args=(fullfile,))
    t.start()

currxlen = 0
currylen = 0
c = 0
tc = 0
chanmodel = 0
preds = []

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)


        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        #print(args)


        #self.x = list(range(100))  # 100 time points
        #self.y = [randint(0, 100) for _ in range(100)]  # 100 data points
        self.x = []
        self.y = []



        self.data_line = self.graphWidget.plot(self.x, self.y)
        #self.win = pg.GraphicsWindow()

        #self.data_line1 = self.win.addPlot(self.y, row=0, col=0)
        #self.data_line2 = self.win.addPlot(self.y, row=0, col=1)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.updateplot)
        self.timer.start()


    def updateplot(self):
        global channelsdict
        global currxlen
        global currylen
        global c
        global chanmodel
        global preds
        global timeout
        global dat_lock
        plottimeinterval = 0.500
        step1 = int(1000*plottimeinterval)
        xran = 5000
        #self.x = self.x[1:]
        #self.x.append(self.x[-1]+1)

        #self.y = self.y[1:]
        #self.y.append(randint(0,100))
        if(channelsdict and chanmodel!=0):

            #if len(channelsdict['time']) == len(channelsdict['preds']):
            if len(channelsdict['time']) > timeslice and len(channelsdict['preds']) > timeslice:
            #if True:
                #appdlen = len(channelsdict['time'])-currxlen
                #appdlen = len(channelsdict['time'])-c

                #print(appdlen)
                #self.tempx = channelsdict['time'][currxlen:currxlen + appdlen]
                #self.tempy = channelsdict['4'][currylen:currylen + appdlen]

                dat_lock.acquire()

                self.tempx = channelsdict['time']
                #self.tempy = channelsdict['4']
                self.tempy = channelsdict['preds']


                dat_lock.release()

                if len(self.tempy) < len(self.tempx):
                    lagging = len(self.tempy)
                else:
                    lagging = len(self.tempx)
                appdlen = lagging - c
                #print('tempx: ',len(self.tempx),'tempy: ', len(self.tempy), 'lagging:',lagging, 'applen: ',appdlen, 'c: ',c)
                if appdlen>0:
                    if appdlen<step1:
                        tempstep = appdlen
                    else:
                        tempstep = int(step1)
                    for f in range(tempstep):
                        self.x.append(self.tempx[c])
                        self.y.append(self.tempy[c])
                        c += 1
                        # self.x.append(self.x[-1]+1)
                    if len(self.x) > xran:
                        self.x = self.x[int(tempstep):]
                        self.y = self.y[int(tempstep):]
                        self.graphWidget.setXRange((c-xran-1)*0.001, (c+5)*0.001, padding=0)

                        # self.y.append(self.y[-1]+1)

                    #self.graphWidget.setYRange(-100, 100, padding=0) #for waveform
                    self.graphWidget.setYRange(-1, 2, padding=0)    #for prediction
                    if len(self.x) == len(self.y):
                        self.data_line.setData(self.x, self.y)

                    global tc
                    tc += 1
                    #currxlen = len(channelsdict['time'])
                    #currylen = len(channelsdict['time'])
                    #currxlen = len(channelsdict['time'])+c
                    #currylen = len(channelsdict['time'])+c


        if timecount>5:
            timedout = True

            self.close()
            #self.join()
            exit(0)





def plot_live():
    print('started live plotting')
    global t1
    t1.start() # zhu prog
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
predlist = []
def make_prediction(sqzedrespd):
    global predlist
    currspect = eg.get_spectrogram(sqzedrespd)
    preds_lock.acquire()
    predlist.append(np.argmax(chanmodel.predict(tf.expand_dims([currspect], -1)), axis=1)[0])
    preds_lock.release()

#overlap = 0.75 #4 slices per second
overlap= 0.67 # 3 slices per second
#overlap = 0.80 # 5 slices per second
#overlap = 0 # no overlap
timeslice = 1000
def RealTime(filename, cm):
    global chanmodel
    global preds_lock
    global currpred
    global predlist
    chanmodel = cm
    print('starting realtime')
    global timecount
    timecount = 0
    stepsize = int((timeslice / filerate) * 8)
    datcount2 = 0
    pointer = 0
    timedout = False
    chand = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'time':[],'preds':[]}
    global channelsdict
    preds = []
    step = (1 - overlap) * timeslice
    slices = int(1/(1-overlap))
    buffer = int((1 / (1 - overlap) - 1) * step)
    totslice = timeslice + buffer
    t2 = threading.Thread(target=plot_live, args=())
    t2.start()
    predictorlist = []

    predictionThreads = []

    while not timedout:
        if exists(filename) and os.stat(filename).st_size > 0:
            with open(filename, "r") as f:
                currfile = pd.read_csv(f, encoding="Shift-JIS", sep='\t')
            currcount = len(currfile)
            #print('currcount-datcount2',currcount-datcount2, datcount2,currcount - datcount2 < (timeslice*8),currcount - datcount2 < (totslice*8))
            #print(currfile)


            if currcount - datcount2 < (int(timeslice)*8):
                time.sleep(0.2)
                #await asyncio.sleep(0.2)
                timecount += 0.2
            else:
                timecount = 0
                #print('reading succ')

                for k in range(int(timeslice)):

                    chand['time'].append(currfile['Time'][datcount2])

                    for i in range(8):
                        #print(currfile['Ch'][datcount2 + i])
                        chand[str(int(currfile['Ch'][datcount2 + i]))].append(currfile['Value'][datcount2 + i])
                    datcount2 += 8

                resamplesiz = int(timeslice / 1000 * 1024)

                #print(datcount2, slices, step)
                #if currcount - datcount2 > (int(totslice) * 8):
                #print('buffer:',datcount2/8 - pointer, 'totslice:',totslice,'timeslice:',timeslice, 'pointer:',pointer)
                if datcount2/8 - pointer > totslice:

                    for i in range(slices):
                        curstep = int((slices-(i+1))*step)
                        #wavtens = tf.constant(chand['1'][int(datcount2/8) - (timeslice+curstep):int(datcount2/8)-curstep])
                        wavtens = tf.constant(chand['1'][pointer:pointer+timeslice])
                        #print(i, 'pointer:',pointer,pointer+timeslice,'curstep:',curstep,'channellengths:', len(chand['1']),len(chand['time']), chand['time'][len(chand['time'])-1])
                        #print(i, int(datcount2/8) - (timeslice+curstep),curstep, len(chand['1']), chand['time'][len(chand['time'])-1])
                        wavtens = signal.resample(wavtens, resamplesiz)  # 1024 for every 1000
                        sqzedrespd = tf.squeeze(wavtens)
                        #print(i, sqzedrespd)

                        predictorthread = threading.Thread(target=make_prediction, args=([sqzedrespd]))
                        predictorlist.append(predictorthread)
                        predictorthread.start()

                        pointer += curstep

                    for s in predictorlist:
                        s.join()
                    preds_lock.acquire()
                    currpred = int(round(np.mean(predlist)))
                    print(chand['time'][len(chand['time'])-1], currpred, predlist)
                    predlist = []
                    preds_lock.release()

                    for s in range(int(timeslice)):
                        chand['preds'].append(currpred)


                    # print(len(channelsdict))
                    #print('readdat', datcount2)
                    #print(channelsdict)
        dat_lock.acquire()
        #print('here',len(chand['time']),len(chand['preds']))
        channelsdict = chand
        dat_lock.release()

        if timecount>5:
            timedout = True
            break





    print('End Realtime')
    plt.plot(range(len(chand['1'])), chand['1'])
    plt.savefig('temptest.png')


    global tc
    print('totco',tc)


