"""Recurrent Neural Network."""

# Part 1 - Data Preprocessing
# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing Training Set
dataset_train = pd.read_csv("dataset_train.csv")

training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train, y_train = [], []

for i in range(60, 1258):

    X_train.append(training_set_scaled[i - 60: i, 0])

    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train,
                     (X_train.shape[0],
                      X_train.shape[1],
                      1))


# Part 2 - Building RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Initialising RNN
regressor = Sequential()

# Adding the first LSTM layer and then Dropout regularising
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))

# Adding the second LSTM layer and then Dropout regularising
regressor.add(LSTM(units = 50,
                   return_sequences = True))

regressor.add(Dropout(0.2))

# Adding the third LSTM layer and then Dropout regularising
regressor.add(LSTM(units = 50,
                   return_sequences = True))

regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and then Dropout regularising
regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling RNN
regressor.compile(optimizer = "adam",
                  loss = "mean_squared_error")

# Fitting RNN to Training Set
regressor.fit(X_train,
              y_train,
              epochs = 100,
              batch_size = 32)
