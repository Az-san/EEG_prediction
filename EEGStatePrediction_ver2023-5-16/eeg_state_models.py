import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from preprocess_eeg import *

class eeg_neural:
    def __init__(self, labelnames, epochs):
        self.label_names = labelnames
        self.preprocessor = preprocssor(labelnames)
        self.epochs = epochs

    def build_the_model(self, train_spectrogram_ds, val_spectrogram_ds, test_spectrogram_ds):
        self.train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(len(train_spectrogram_ds)).prefetch(tf.data.AUTOTUNE)
        self.val_spectrogram_ds = val_spectrogram_ds.cache().shuffle(len(val_spectrogram_ds)).prefetch(tf.data.AUTOTUNE)
        self.test_spectrogram_ds = test_spectrogram_ds.cache().shuffle(len(test_spectrogram_ds)).prefetch(tf.data.AUTOTUNE)

        for example_spectrograms, example_spect_labels in self.train_spectrogram_ds.take(1):
            break
        input_shape = example_spectrograms.shape[1:]
        print('Input shape:', input_shape)
        num_labels = len(self.label_names)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = tf.keras.layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=self.train_spectrogram_ds.map(map_func=lambda spec, label: spec))

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        self.model.summary()
    
    def compile_the_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
    
    def train_the_model(self):
        EPOCHS = self.epochs
        self.history = self.model.fit(
            self.train_spectrogram_ds,
            validation_data=self.val_spectrogram_ds,
            epochs=EPOCHS,
            #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )
    
    def evaluate(self):
        print(self.model.evaluate(self.test_spectrogram_ds, return_dict=True))
        self.y_pred = self.model.predict(self.test_spectrogram_ds)
        self.y_pred = tf.argmax(self.y_pred, axis=1)
        self.y_true = tf.concat(list(self.test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
        #self.get_num_lines(filename)-8*self.timeslice

    def make_prediction(self,wave):
        #print('predictionwave',type(wave),wave.shape)
        currspect = self.preprocessor.convert_to_spectrogram(wave)
        #print('prediction spectrogramshape',currspect.shape)
        #self.preprocessor.display_spectrogram(wave,currspect,'prediction')
        return np.argmax(self.model.predict(np.array([currspect]))[0])

    def run_inference(self, wave):
        x = wave
        x = self.preprocessor.convert_to_spectrogram(x)
        x = x[tf.newaxis,...]

        prediction = self.model(x)
        x_labels = self.label_names
        plt.bar(x_labels, tf.nn.softmax(prediction[0]))
        plt.title('Inference')
        plt.show()

        #display.display(display.Audio(wave, rate=16000))

    def plot_metrics(self):
        self.metrics = self.history.history
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(self.history.epoch, self.metrics['loss'], self.metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Loss [CrossEntropy]')

        plt.subplot(1,2,2)
        plt.plot(self.history.epoch, 100*np.array(self.metrics['accuracy']), 100*np.array(self.metrics['val_accuracy']))
        plt.legend(['accuracy', 'val_accuracy'])
        plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')
        plt.show()
    
    def plot_confusion_matrix(self):
        confusion_mtx = tf.math.confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx,
                    xticklabels=self.label_names,
                    yticklabels=self.label_names,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()