import sys
import configparser
import os
import logging
import datetime
import time
import tkinter as tk
from tkinter import filedialog

time_slice = 1024
epochs = 50
num_labels = 2
modelname = 'testmodel'
iterations = 1
overlap = 0.5
decTrain = 0.6
make = 'create'

def remove_oldest(maxfiles):
    DIR = 'logs/'
    files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    numfiles = len(files)
    #print(files)

    while(numfiles>maxfiles):
        os.remove(DIR+files[0])
        logging.warning('Removed oldest log file: '+DIR+files[0]+', as logs in directory exceeded maxfiles: '+str(maxfiles)+'.')
        #print('Removed '+DIR+files[0])
        files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))] # Update files list
        numfiles = len(files)   # Update number of files
        #delete oldest

def argparser(args): #Unused
    logging.warning('Running argparser')
    #print('args: ', args)
    help = 'Welcome to EEG Eye Open/Closed Evaluator! \n' \
           'Valid arguments are <-S model> [-S name] [-i time] [-i epochs] [-i labels] [-i iterations].\n\n' \
           'Where:\n' \
           '<-String model> Takes values "create" or "load", to opt to create new model or load existing model,\n' \
           '[-String name] Name newly created models (cannot contain spaces) {"testmodel"},\n' \
           '[-integer time] Time slice length (ms) {1024},\n' \
           '[-integer epochs] Training epochs per channel per iteration {50},\n' \
           '[-integer labels] Number of classifications {2},\n' \
           '[-integer iterations] Experiment iterations per channel training model {10}.\n\n' \
           '<required argument> [optional argument] {default value}\n\n' \
           'For Example:\n' \
           '"python EEGData.py create newmodel 3000 100 2 30"\n' \
           '"python EEGData.py load"\n'
    if len(args) < 2 or len(args)>7:
        print('Please refer to "main.py -help" for valid arguments.')
        if len(args) == 1:
            print('No arguments were given.')
        elif len(args) > 7:
            print('Excessive arguments were given.')
        exit(0)
    else:
        if str(args[1]) == 'create':
            make = 'create'
        elif str(args[1]) == 'load':
            make = 'load'
        elif str(args[1]) == '-help':
            print(help)
            exit(0)
        else:
            print('Invalid starting argument. Please refer to "main.py -help" for valid arguments.',args[1])
            exit(0)

        if len(args)>2:
            modelname = str(args[2])
        if len(args)>3:
            if args[3].isnumeric():
                global time_slice
                time_slice = int(args[3])
            else:
                print('Invalid argument for time slice (not numerical):',time_slice)
                exit(0)
        if len(args)>4:
            if args[4].isnumeric():
                global epochs
                epochs = int(args[4])
            else:
                print('Invalid argument for epochs (not numerical):', epochs)
                exit(0)
        if len(args)>5:
            if args[5].isnumeric():
                global num_labels
                num_labels = int(args[5])
            else:
                print('Invalid argument for number of labels (not numerical):', num_labels)
                exit(0)
        if len(args)>6:
            if args[6].isnumeric():
                global iterations
                iterations = int(args[6])
            else:
                print('Invalid argument for iterations (not numerical):', iterations)
                exit(0)
    return make,modelname,time_slice,epochs,num_labels,iterations

config = configparser.ConfigParser()
def write_configfile(mn, ts, ol, nl, dt, ep, it):
    config['GENERAL'] = {'modelname': str(mn),
                         'time_slice': str(ts)}
    config['TRAINING'] = {'overlap': str(ol),
                          'num_labels': str(nl),
                          'decTrain': str(dt),
                          'epochs': str(ep),
                          'iterations': str(it)}
    with open('EEGData.properties', 'w') as configfile:
        config.write(configfile)

def read_configfile():

    if(not os.path.exists('EEGData.properties')):
        logging.warning('read_configfile file does not exists and therefore will be written with default values')
        write_configfile(modelname, time_slice, overlap, num_labels, decTrain, epochs, iterations)

    config.read('EEGData.properties')
    try:
        mn = str(config['GENERAL']['modelname'])
        ts = int(config['GENERAL']['time_slice'])
        ol = float(config['TRAINING']['overlap'])
        nl = int(config['TRAINING']['num_labels'])
        dt = float(config['TRAINING']['decTrain'])
        ep = int(config['TRAINING']['epochs'])
        it = int(config['TRAINING']['iterations'])
    except:
        logging.warning('read_configfile file contains invalid values and will be re-written with default values')
        write_configfile(modelname, time_slice, overlap, num_labels, decTrain, epochs, iterations)
    mn = str(config['GENERAL']['modelname'])
    ts = int(config['GENERAL']['time_slice'])
    ol = float(config['TRAINING']['overlap'])
    nl = int(config['TRAINING']['num_labels'])
    dt = float(config['TRAINING']['decTrain'])
    ep = int(config['TRAINING']['epochs'])
    it = int(config['TRAINING']['iterations'])

    return mn, ts, ol, nl, dt, ep, it




def guimenu():
    tmn, tts, tol, tnl, tdt, tep, tite = read_configfile()
    logging.warning('Started Options Selection GUI Menu.')
    top = tk.Tk(className='EEG Options Selection')
    top.geometry('300x300')
    createframe = tk.LabelFrame(top, text="Create Model Options")
    loadframe = tk.LabelFrame(top, text="Load Model Options")
    createb = tk.Button(top, text='Submit')
    loadb = tk.Button(top, text='Submit')

    def createoptions(mn, ts, ol, nl, dt, ep, it):
        #print('entered options')
        global time_slice
        time_slice = int(ts)
        global epochs
        epochs = int(ep)
        global num_labels
        num_labels = int(nl)
        global modelname
        modelname = str(mn)
        global iterations
        iterations = int(it)
        global overlap
        overlap = float(ol)
        global decTrain
        decTrain = float(dt)
        write_configfile(mn, ts, ol, nl, dt, ep, it)
        top.destroy()
        logging.warning('Creating new model with the following configurations:')
        logging.warning('Model Name: ' + str(mn))
        logging.warning('Time Slice (ms): ' + str(ts))
        logging.warning('Overlap Ratio: ' + str(ol))
        logging.warning('Classifications: ' + str(nl))
        logging.warning('Dataset Ratios: ' + str(dt))
        logging.warning('Epochs: '+str(ep))
        logging.warning('Experiment Iterations: ' + str(it))

    def loadoptions(mn, ts):
        global modelname
        modelname = str(mn)
        global time_slice
        time_slice = int(ts)
        write_configfile(mn, ts, tol, tnl, tdt, tep, tite)
        logging.warning('Loading existing model with the following configurations:')
        logging.warning('Model Name: ' + str(mn))
        logging.warning('Time Slice (ms): ' + str(ts))
        top.destroy()


    var1 = tk.StringVar(createframe)
    var1.set(str(tts))

    def choose_create():
        #print('create')
        global make
        make = 'create'
        loadframe.pack_forget()
        loadb.pack_forget()
        #labelframe.pack(fill="both", expand="yes")
        createframe.pack()
        L1 = tk.Label(createframe, text="Model Name: ")
        E1 = tk.Entry(createframe, bd=5)
        E1.insert(0,str(tmn))
        L2 = tk.Label(createframe, text="Time Slice Length (ms): ")
        E2 = tk.Spinbox(createframe, bd=5, from_=10, to=10000, width=18, textvariable=var1)
        L3 = tk.Label(createframe, text="Overlap Ratio: ")
        var3 = tk.StringVar(createframe)
        var3.set(str(tol))
        E3 = tk.Spinbox(createframe, bd=5, from_=0.0, to=1.0, width=18, increment=0.05,textvariable=var3)
        L4 = tk.Label(createframe, text="Classifications:")
        var4 = tk.StringVar(createframe)
        var4.set(str(tnl))
        E4 = tk.Spinbox(createframe, bd=5, from_=2, to=5, width=18,textvariable=var4)

        L5 = tk.Label(createframe, text="Training Set Ratio:")
        var5 = tk.StringVar(createframe)
        var5.set(str(tdt))
        E5 = tk.Spinbox(createframe, bd=5, from_=0.0, to=1.0, width=18, increment=0.1,textvariable=var5)

        L6 = tk.Label(createframe, text="Epochs:")
        var6 = tk.StringVar(createframe)
        var6.set(str(tep))
        E6 = tk.Spinbox(createframe, bd=5, from_=10, to=2000, width=18,textvariable=var6)

        L7 = tk.Label(createframe, text="Experiment Iterations:")
        var7 = tk.StringVar(createframe)
        var7.set(str(tite))
        E7 = tk.Spinbox(createframe, bd=5, from_=1, to=100, width=18,textvariable=var7)

        L1.grid(row=0,column=0)
        E1.grid(row=0,column=1)
        L2.grid(row=1,column=0)
        E2.grid(row=1,column=1)
        L3.grid(row=2,column=0)
        E3.grid(row=2,column=1)
        L4.grid(row=3,column=0)
        E4.grid(row=3,column=1)
        L5.grid(row=4,column=0)
        E5.grid(row=4,column=1)
        L6.grid(row=5,column=0)
        E6.grid(row=5,column=1)
        L7.grid(row=6, column=0)
        E7.grid(row=6, column=1)
        createb.configure(command=lambda:[createoptions(E1.get(), E2.get(), E3.get(), E4.get(), E5.get(), E6.get(), E7.get())])
        createb.pack()


    def choose_load():
        #print('load')
        global make
        make = 'load'
        # labelframe.pack(fill="both", expand="yes")
        createframe.pack_forget()
        createb.pack_forget()
        loadframe.pack()
        L1 = tk.Label(loadframe, text="Model Name: ")
        E1 = tk.Entry(loadframe, bd=5)
        E1.insert(0, str(tmn))
        L2 = tk.Label(loadframe, text="Time Slice Length (ms): ")
        E2 = tk.Spinbox(loadframe, bd=5, from_=10, to=10000, width=18, textvariable=var1)
        L1.grid(row=0, column=0)
        E1.grid(row=0, column=1)
        L2.grid(row=1, column=0)
        E2.grid(row=1, column=1)
        loadb.configure(command=lambda:[loadoptions(E1.get(), E2.get())])
        loadb.pack()

    # Code to add widgets will go here...
    #b = tk.Button(top, text='lol',command=choose_load)
    #b.pack(padx=100,pady=50)
    mb = tk.Menubutton(top, text="Model", relief=tk.RAISED)
    mb.grid()
    mb.menu = tk.Menu(mb, tearoff=0)
    mb["menu"] = mb.menu
    #mb.menu.invoke()
    #print('adding buttons')

    mb.menu.add_radiobutton(label="Create",command=choose_create)
    mb.menu.add_radiobutton(label="Load",command=choose_load)
    mb.pack()
    #aaa = tk.Entry(top, bd=5)
    #aaa.pack()
    #bbb = tk.Button(top, text='Submit', command=lambda:[print(aaa.get())])
    #bbb.pack()
    top.mainloop()
    return time_slice, epochs, num_labels, modelname, iterations, overlap, decTrain, make

def getfile():
    #logging.warning('Started File Selection GUI Menu.')
    root = tk.Tk()
    #root.withdraw()
    
    
    file_path = filedialog.askopenfilename()
    filename = str(file_path)
    root.destroy()
    return filename