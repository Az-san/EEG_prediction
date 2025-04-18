# welcome to Real-Time EEG reader!
# this is a simple version of the single channel EEG model trainer and predictor
from eeg_reader import *
from preprocess_eeg import *
from eeg_state_models import *
import matplotlib.pyplot as plt
from export_eeg_model import *
from eeg_select_options import *
from eeg_realtime_prediction import *

# default parameters
relevant_channels = [1] # keep only one value inside
traintsvfilename=''
tsvtrainfilename = getfile()
laptop = False
print(tsvtrainfilename)
tsvtrainfilename = tsvtrainfilename.replace('/', '\\')
print(tsvtrainfilename)
# if traintsvfilename=='':
#     tsvtrainfilename = 'C:\\Users\\Akio Wakata\\PycharmProjects\\EEGDataAnalysis\\Natsume(Pz) Auto 2021-12-03 16-02-03.901.tsv' # tsv files only
# #tsvfilename = 'C:\\ExperimentationRelated12-2\\EEGStatePrediction\\Natsume(Pz) Auto 2021-12-03 16-04-23.989.tsv'
# if laptop:
#     tsvtrainfilename = 'D:\\EEGStatePrediction\\Natsume(Pz) Auto 2021-12-03 16-02-03.901.tsv'

numlabels = 2
label_names = ['Open','Closed']
training_ratio = 0.6
time_slice = 3000
overlap = 0.20
epochs = 100
prediction_interval = 1000

print('welcome to EEG reader')

# read the training data
eeg_data_reader = read_file(tsvtrainfilename, relevant_channels)
single_file_single_channel_eeg = eeg_data_reader.read()[:60000]

print(single_file_single_channel_eeg)
# preprocess the training data
eeg_data_preprocessor = preprocssor(label_names)
sliced_waves = eeg_data_preprocessor.divide_time_timeslices(single_file_single_channel_eeg, time_slice, overlap)
print(sliced_waves.shape)
classed_waves, labels = eeg_data_preprocessor.divide_into_classes(sliced_waves, numlabels)
print('class labels shape',classed_waves.shape,labels.shape)
trainset, valset, testset = eeg_data_preprocessor.define_datasets(classed_waves, labels, numlabels, training_ratio)

# for example_audio, example_labels in trainset.take(1):  
    
#     print('examplespec',example_audio[0].shape)
#     print('examplelabels',example_labels[0])
#exit(0)

eeg_nn_model = eeg_neural(label_names, epochs)
eeg_nn_model.build_the_model(trainset, valset, testset)
eeg_nn_model.compile_the_model()
eeg_nn_model.train_the_model()



# print('doing model2')
# eeg_data_preprocessor2 = preprocssor(label_names)
# eeg_data_reader2 = read_file(tsvtrainfilename, [2])
# single_file_single_channel_eeg2 = eeg_data_reader2.read()[:60000]
# sliced_waves2 = eeg_data_preprocessor2.divide_time_timeslices(single_file_single_channel_eeg2, time_slice, overlap)
# classed_waves2, labels2 = eeg_data_preprocessor2.divide_into_classes(sliced_waves2, numlabels)
# trainset2, valset2, testset2 = eeg_data_preprocessor2.define_datasets(classed_waves2, labels2, numlabels, training_ratio)
# eeg_nn_model2 = eeg_neural(label_names, epochs)
# eeg_nn_model2.build_the_model(trainset2, valset2, testset2)
# eeg_nn_model2.compile_the_model()
# eeg_nn_model2.train_the_model()


eeg_nn_model.plot_metrics()
eeg_nn_model.evaluate()
eeg_nn_model.plot_confusion_matrix()

# eeg_nn_model2.plot_metrics()
# eeg_nn_model2.evaluate()
# eeg_nn_model2.plot_confusion_matrix()
# eeg_nn_model.run_inference(sliced_waves[0])
# eeg_nn_model.run_inference(sliced_waves[1])

# pred = eeg_nn_model.make_prediction(single_file_single_channel_eeg[:1000])
# print(pred)
# exit(0)


#export save
# export = ExportModel(eeg_nn_model.model)
#export(tf.constant(str(data_dir/'no/01bb6a2a_nohash_0.wav')))

#save reload
# tf.saved_model.save(export, "saved")
# imported = tf.saved_model.load("saved")
# imported(waveform[tf.newaxis, :])

# Do realtime
tsvtestfilename = getfile()
#tsvtestfilename = 'C:\\Users\\Akio Wakata\\PycharmProjects\\EEGDataAnalysis\\Natsume(Pz) Auto 2021-12-03 16-02-03.901.tsv' # tsv files only
# if laptop:
#     tsvtestfilename = 'D:\\EEGStatePrediction\\Natsume(Pz) Auto 2021-12-03 16-02-03.901.tsv'
realtime_predictor = eeg_realtime(label_names,eeg_nn_model.model, time_slice,relevant_channels, prediction_interval)
realtime_predictor.real_time(tsvtestfilename)


#single_file_single_channel_spectrogram = eeg_data_preprocessor.audio_to_spectrogram(single_file_single_channel_eeg[start:end])

#print('Waveform shape:', single_file_single_channel_eeg.shape)
# eeg_data_preprocessor.display_spectrogram(eeg_data_preprocessor.train_dat[0], np.array(list(trainset.as_numpy_iterator())[0][0]), 'test')

# print(eeg_data_preprocessor.train_dat[0][:20])
# print(single_file_single_channel_eeg[1:20], list(eeg_data_preprocessor.train_dat[0]) == list(single_file_single_channel_eeg[1:1001]))
#print(classed_waves.shape, classed_waves)


 #test spectrogram visual
# for i in range(59):
#     start = i*1000
#     end = start+1000
#     if (start>0 and start<15000) or (start>30000 and start<45000):
#         label = 'OPEN'
#     else:
#         label = 'CLOSED'
#     single_file_single_channel_spectrogram = eeg_data_preprocessor.convert_to_spectrogram(single_file_single_channel_eeg[start:end])
#     #single_file_single_channel_spectrogram = eeg_data_preprocessor.audio_to_spectrogram(single_file_single_channel_eeg[start:end])

#     print('Waveform shape:', single_file_single_channel_eeg.shape)
#     print('Spectrogram shape:', single_file_single_channel_spectrogram.shape)
#     eeg_data_preprocessor.display_spectrogram(single_file_single_channel_eeg[start:end], single_file_single_channel_spectrogram, label+': '+str(start)+'-'+str(end))
    
# 32
# train the model


