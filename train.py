import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import tensorflow as tf

from classification_models.tfkeras import Classifiers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import roc_curve, auc


def get_model(model_name, input_shape=(80,80,1)):
#   ClsModel, preprocess_input = Classifiers.get(model_name)
  ClsModel, _ = Classifiers.get(model_name)

  # X = preprocess_input(X_train)
  # Xt = preprocess_input(X_test)

  # build model
  base_model = ClsModel(input_shape=input_shape, include_top=False)
  x = GlobalAveragePooling2D()(base_model.output)
  output = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=[base_model.input], outputs=[output])
  return model

def train_model(model_name, X_train, y_train, X_test, y_test, input_shape=(80, 80, 1),  batch_size=32, epochs=50):
    model_path = 'models/' + model_name + '.h5'
    print(model_name, ':', 'model at path', model_path)
    
    model = get_model(model_name, input_shape=input_shape)

    print(model_name, ':', 'compiling /w RMSprop')
    model.compile(optimizer=RMSprop, loss='binary_crossentropy', metrics=['accuracy'])
    print(model_name, ':', 'fitting /w RMSprop')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=batch_size, 
              epochs=epochs, 
              callbacks=[tensorboard_callback])

    print(model_name, ':', 'compiling /w SGD')
    model.compile(optimizer=SGD, loss='binary_crossentropy', metrics=['accuracy'])
    print(model_name, ':', 'fitting /w SGD')
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=batch_size, 
              epochs=epochs, 
              callbacks=[tensorboard_callback])

    model.save(model_path)
    print(model_name, ':', 'saved')
    
    return model

def benchmark(model, filename='result.pickle', save=True):
  y_pred = model.predict(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  auc_score = auc(fpr, tpr)
  print('AUC:', auc_score)
  result = {
      "auc": auc_score,
      "fpr": fpr,
      "tpr": tpr,
      "thresholds": thresholds
  }
  if save:
    with open(filename, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

  plt.plot(fpr, tpr)
  plt.title("ROC Curve (%s)" % (filename,))
  plt.show()


data_path = 'data/'
with open(data_path + 'split_data.pickle', 'rb') as fp:
    data = pickle.load(fp)
    
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

RMSprop = optimizers.RMSprop(learning_rate=1E-3, rho=0.9, epsilon=1E-8)
lr = optimizers.schedules.ExponentialDecay(1E-3, 1, 0.92, staircase=True)
SGD = optimizers.SGD(learning_rate=lr, momentum=0.0)

resize_method = 'duplicate'
ch = 3

if resize_method == 'duplicate':
    train_shape = tuple(list(X_train.shape[: -1]) + [ch])
    X_train_new = np.broadcast_to(X_train, train_shape).copy()
    
    test_shape = tuple(list(X_test.shape[: -1]) + [ch])
    X_test_new = np.broadcast_to(X_test, test_shape).copy()
    
    print(X_train_new.shape, X_test_new.shape)

model_name = 'inceptionresnetv2'
model = train_model(model_name, X_train_new, y_train, X_test_new, y_test, input_shape=(80, 80, 3), batch_size=32, epochs=50)
benchmark(model, filename='results/' + model_name + '.pickle', save=True)
