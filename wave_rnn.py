import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import math

def get_data(df, n_steps):
    """
        inputs:
            df, a pandas dataframe containing features and target
            n_steps, an int representing the number of previous days 
                to use to predict each subsequent day
        outputs:
            X_train, X_test, Y_train, numpy arrays containing testing 
                and training data
            scaler, an sklearn MinMaxScaler used to normalize the data
    """
    data = df.drop('Date/Time', axis = 1).to_numpy()

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    n_features = data.shape[1]
    n_samples = len(data) - n_steps

    # split data into X (windows with n_steps days each) and y (next day heights)
    X = np.zeros((n_samples, n_steps, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = data[i:i + n_steps]
        y[i] = data[i + n_steps][0]

    split_index = int(n_samples * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = y[:split_index], y[split_index:]

    return X_train, X_test, Y_train, scaler

def evaluate(model, X_train, X_test, df, n_steps, scaler):
    """
        inputs:
            model,
            X_train, X_test,
            df,
            n_steps,
            scaler,
        outputs:
            none, simply prints the training and testing scores
    """
    predicted_train = model.predict(X_train)
    predicted_test = model.predict(X_test)

    a = np.zeros((len(predicted_train), 6))
    b = np.zeros((len(predicted_test), 6))

    a[:,0]=predicted_train.reshape(X_train.shape[0])
    b[:,0]=predicted_test.reshape(X_test.shape[0])

    # un-normalize predictions
    predicted_train = scaler.inverse_transform(a)
    predicted_test = scaler.inverse_transform(b)

    # get original data values
    y_train_raw = df['Hs'][n_steps:n_steps+X_train.shape[0]].values
    y_test_raw = df['Hs'][n_steps+X_train.shape[0]:].values

    # calculate mean squared error
    trainScore = mean_squared_error(y_train_raw, predicted_train[:,0])
    print('Training rMSE: %.2f meters' % (math.sqrt(trainScore)))
    testScore = mean_squared_error(y_test_raw, predicted_test[:,0])
    print('Testing rMSE: %.2f meters' % (math.sqrt(testScore)))


def main():

    # read in csv and fill in missing values
    df = pd.read_csv('Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv').iloc[2:]
    df = df.replace(-99.9,np.nan)
    df = df.interpolate()

    # split df into numpy arrays for testing and training
    windowsize = 250
    X_train, X_test, Y_train, scaler = get_data(df, windowsize)

    # instantiate and compile model
    n_units = 25
    n_epochs = 1
    batch_size = 250

    rnn_model = keras.Sequential([
        keras.layers.LSTM(n_units),
        keras.layers.Dense(1)
        ])

    rnn_model.compile(
        loss='mean_squared_error',
        optimizer='adam'
        )

    # fit model
    rnn_model.fit(
        X_train,
        Y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_split = 0.1
    )

    # test model
    evaluate(rnn_model, X_train, X_test, df, windowsize, scaler)

if __name__ == '__main__':
    main()