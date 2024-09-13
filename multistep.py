import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,GRU,Bidirectional,InputLayer

def create_multistep_model(sequence_length, n_features, units, n_layers, dropout, k):
    model = Sequential()
    
    # Input layer
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, n_features)))
    
    # Add additional LSTM layers
    for _ in range(n_layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
    
    # Add a final LSTM layer
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    
    # Output layer with 'k' units for multistep prediction
    model.add(Dense(units=k))  
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def prepare_multistep_data(data, sequence_length, k):
    X, y = [], []
    for i in range(len(data) - sequence_length - k + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + k])
    return np.array(X), np.array(y)

