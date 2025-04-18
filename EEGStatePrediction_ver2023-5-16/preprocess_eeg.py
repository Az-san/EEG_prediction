# this file converts the raw eeg into usable spectrogram groups

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
batch_size = 64
class preprocssor:
    def __init__(self, labelnames):
        self.labelnames = labelnames

    def divide_into_classes(self, sliced_waves, number_of_classes):
        class_labels = np.array([], dtype=int)
        for i in range(number_of_classes):
            class_labels = np.append(class_labels, np.full((sliced_waves.shape[0]//number_of_classes,),np.array([i])))
        print(class_labels, class_labels.shape)
        
        if sliced_waves.shape[0]%len(self.labelnames)!=0:
            sliced_waves = sliced_waves[sliced_waves.shape[0]%len(self.labelnames):] # remove first element from slices if array of time slices is odd (balances out 0 and 1 labels)
        #print(tf.shape(sliced_waves))
        #classed_waves_dataset = np.column_stack((sliced_waves,class_labels))
        # print(classed_waves_dataset, classed_waves_dataset.shape)
        # print(classed_waves_tensor)
        #classed_waves_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([sliced_waves,class_labels]))
        #classed_waves = sliced_waves.reshape((number_of_classes,sliced_waves.shape[0]//number_of_classes,sliced_waves.shape[1]))
        
        return sliced_waves, class_labels#[:,None]
    
    def divide_time_timeslices(self, wave, time_window_size, overlap):
        #sliced_waves = np.split(clss, )
        if overlap == 0 or overlap == 1:
            stepsize = time_window_size
            num_slices = wave.shape[0]//stepsize
            
        else:
            stepsize = int(time_window_size*overlap)
            num_slices = wave.shape[0]//stepsize-1
        #sliced_waves = np.lib.index_tricks.as_strided(wave,shape=[num_slices,time_window_size], strides=(stepsize*4,1*4)).copy()
        #sliced_waves = np.lib.index_tricks.as_strided(wave,shape=[num_slices,time_window_size], strides=wave.strides*2).copy()
        sliced_waves = np.lib.index_tricks.as_strided(wave,shape=(wave.size-time_window_size+1,time_window_size), strides=wave.strides*2)[0::stepsize].copy()
        
        # print('here')
        # print(wave[0:5])
        # print(sliced_waves[0,:5])
        

        return np.array(sliced_waves)
    
    def define_datasets(self, all_waves, all_labels, numlabels, ratio_of_training_set_to_1):
        len_training = all_waves.shape[0]*ratio_of_training_set_to_1 
        len_valtest = (all_waves.shape[0]-len_training)//2
        
        # print('defining',len(all_waves),all_labels,int(len(all_labels)/numlabels))
        # labels_by_label = np.split(all_labels, numlabels)
        # print(np.array(all_waves).shape)
        # wavs_by_label = np.split(np.array(all_waves), numlabels)
        # print(np.array(wavs_by_label).shape, np.array(labels_by_label).shape)
        
        # print('defining2',len(all_waves),labels_by_label)
        
        #divide into training, validation and testing sets (divide among labels evenly)
        #labels
        train_labs, val_labs, test_labs = np.split(np.array(np.split(all_labels, numlabels)), [int(len_training//numlabels), int(len_training//numlabels+len_valtest//numlabels+1)], axis=1) # 
        train_labs, val_labs, test_labs = tf.constant(np.concatenate(np.array(train_labs),axis=0)), tf.constant(np.concatenate(np.array(val_labs),axis=0)), tf.constant(np.concatenate(np.array(test_labs),axis=0))
        #waves
        train_dat, val_dat, test_dat = np.split(np.array(np.split(all_waves, numlabels)), [int(len_training//numlabels), int(len_training//numlabels+len_valtest//numlabels+1)], axis=1) # 
        self.train_dat, val_dat, test_dat = tf.constant(np.concatenate(np.array(train_dat),axis=0)), tf.constant(np.concatenate(np.array(val_dat),axis=0)), tf.constant(np.concatenate(np.array(test_dat),axis=0))
        print('nplabs trainshape',train_labs.shape,'npwaves tainshape',self.train_dat.shape)
        print('nplabs valshape',val_labs.shape,'npwaves valshape',val_dat.shape)
        print('nplabs testshape',test_labs.shape,'npwaves tainshape',test_dat.shape)
        # convert sets to tensorflow dataset
        
        #a = np.array([zipper(x, y) for x,y in (self.train_dat,train_labs)])
        #print('type',type(self.train_dat)    
        train_ds = tf.data.Dataset.from_tensor_slices((self.train_dat, train_labs))
        val_ds = tf.data.Dataset.from_tensor_slices((val_dat, val_labs))
        test_ds = tf.data.Dataset.from_tensor_slices((test_dat, test_labs))
        print('trainds elementspec',train_ds.element_spec)
        print('valds elementspec',val_ds.element_spec)
        print('testds elementspec',test_ds.element_spec)
        #print('tfshape wvae', np.array(list(train_ds.as_numpy_iterator())).shape, np.array(list(train_ds.as_numpy_iterator()))[0][0].shape, np.array(list(train_ds.as_numpy_iterator()))[0][1].shape)
        #labels_by_set = np.split(np.array(np.split(all_labels, numlabels)), [int(len_training//2), int(len_training//2+len_valtest//2+1)], axis=1) # 
        #print('defining3',len(all_waves),labels_by_set)
        # for example_audio, example_labels in train_ds.take(1):  
        #     print('exampleaudio',example_audio[0].shape)
        #     print('examplelabels',self.labelnames[example_labels[0]])
        # print('defining3',len(all_waves),train_labs,len(train_labs), val_labs,len(val_labs), test_labs,len(test_labs))
        # print('defining3',len(all_waves),train_dat.shape, val_dat.shape, test_dat.shape)
        #exit(0)
        #convert dataset waves to spectrograms
        def make_spec_ds(ds):
            return ds.map(
                map_func=lambda wave,label: (self.convert_to_spectrogram(wave), label),
                # map_func=lambda wave,label: (self.convert_to_spectrogram(wave), label),
                num_parallel_calls=tf.data.AUTOTUNE)
        
        # convert each set to spectrogram
        train_spectrogram_ds = make_spec_ds(train_ds)
        val_spectrogram_ds = make_spec_ds(val_ds)
        test_spectrogram_ds = make_spec_ds(test_ds)
        print('trainspects elementspec',train_spectrogram_ds.element_spec)
        print('valspects elementspec',val_spectrogram_ds.element_spec)
        print('testspects elementspec',test_spectrogram_ds.element_spec)

        train_spectrogram_ds = train_spectrogram_ds.batch(batch_size)
        val_spectrogram_ds = val_spectrogram_ds.batch(batch_size)
        test_spectrogram_ds = test_spectrogram_ds.batch(batch_size)
        # print('trainspect elementspec after batch',train_spectrogram_ds.element_spec)
        #print('tfshape spec', np.array(list(train_spectrogram_ds.as_numpy_iterator())).shape)
        #print(train_spectrogram_ds, np.array(list(train_spectrogram_ds.as_numpy_iterator())[0][0]).shape, train_spectrogram_ds.element_spec)
        # print(train_spectrogram_ds, list(train_spectrogram_ds.as_numpy_iterator())[20][0], train_spectrogram_ds.element_spec)
        #exit(0)
        #exit(0)

        #display spectrograms in dataset
        # rows = 5
        # cols = 6
        # n = rows*cols
        # fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

        # for i in range(n):
        #     r = i // cols
        #     c = i % cols
        #     ax = axes[r][c]
        #     #print(train_spectrogram_ds, list(train_spectrogram_ds.take(i+1).as_numpy_iterator())[0][0], train_spectrogram_ds.element_spec)
        #     self.plot_spectrogram(np.array(list(train_spectrogram_ds.as_numpy_iterator())[i][0]), ax)
        #     ax.set_title(self.label_names[np.array(list(train_spectrogram_ds.as_numpy_iterator())[i][1])])
            # print(np.array(list(train_spectrogram_ds.as_numpy_iterator())[i][0][0][:3]), np.array(list(train_spectrogram_ds.as_numpy_iterator())[i][1]))

        # plt.show()

        return train_spectrogram_ds, val_spectrogram_ds, test_spectrogram_ds
        #train_ds = map(return_pair, )
        
    def convert_to_spectrogram(self, wave):
        #spectrogram = tf.signal.stft(wave, frame_length=255, frame_step=128)
        #spectrogram = tf.signal.stft(wave, frame_length=512, frame_step=16, fft_length=512)[:,:25] # good ratio? for 1000ms 0-40hz region
        #spectrogram = tf.signal.stft(wave, frame_length=512, frame_step=16, fft_length=512)[:,2:8] # good ratio? for 1000ms alpha region
        spectrogram = tf.signal.stft(wave, frame_length=1024, frame_step=64, fft_length=1024)[:,:60] # 50 good ratio? for 3000ms and 2000ms
        #print('sha',spectrogram.shape)
        spectrogram = tf.abs(spectrogram) # magnitude of spectrogram
        spectrogram = spectrogram[..., tf.newaxis] # adds 'color channels' dimension to make into image-like data for CNN layers (batch_size, height, width, channels) input shape
        #print('spectrogram shape',spectrogram.shape)
        return spectrogram

    # def audio_to_spectrogram(self, wave):
    #     spectrogram = tfio.audio.spectrogram(wave, 512, 1000, 32)
    #     return spectrogram

    # def make_spec_ds(self,ds):
    #     return ds.map(
    #         map_func=lambda audio,label: (self.convert_to_spectrogram(audio), label),
    #         num_parallel_calls=tf.data.AUTOTUNE)

    def plot_spectrogram(self, spectrogram, ax):
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
        # Convert the frequencies to log scale and transpose, so that the time is
        # represented on the x-axis (columns).
        # Add an epsilon to avoid taking a log of zero.
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

    def display_spectrogram(self, wave, spectrogram, label):
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(wave.shape[0])
        axes[0].plot(timescale, wave)
        axes[0].set_title('Waveform')
        #axes[0].set_xlim([0, 16000])

        # self.plot_spectrogram(spectrogram.numpy()[:,:50], axes[1])
        self.plot_spectrogram(spectrogram, axes[1])
        axes[1].set_title('Spectrogram')
        plt.suptitle(label)
        plt.show()