import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,GRU,Bidirectional,InputLayer

#------------------------------------------------------------------------------
# Create multistep model
# this function defines a multistep forecasting model sing LSTM layers for time series
# prediction 
# the function has parameter like sequence_length which is the nuber of time steps in each sequence
# nfeatures is the number of features at each step
# units is the number of neurons in the LSTM layer 
# dropout is the dropout rate for the LSTM layer
# k is th number of days to predict in the future
#------------------------------------------------------------------------------
def create_multistep_model(sequence_length, n_features, units, n_layers, dropout, k):
    
    #this is to initialize the sequential model. which means layers will be 
    # added one after another in a sequence 

    model = Sequential()
    
    # Input layer
    # the input shape of this layer s defined by the sequence length and the number of features per time step
    # the sequence length is the number of previous time steps used as input for the prediction
    # n_features is the number of variable being used 
    # the return sequence is where the full sequence of optputs from each time step
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, n_features)))
    
    # Add additional LSTM layers
    # for loop is used to add additional LSTM layers based on the value of n_layers
    # the dropout layer is added after each LSTM layer to help prevent overfitting by randomly 
    # setting a fraction of the output units to zero during training.
    for _ in range(n_layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
    
    # Add a final LSTM layer
    # this layer does not use return_sequences=True because it outputs only the final state
    # Dropout is used to reduce overfitting 
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    
    # Output layer with 'k' units for multistep prediction
    # this is a fully connected layer with k units which determines 
    # how many days in the future to predict 
    model.add(Dense(units=k))  
    
    #this is to compile the model 
    #the adam optimizer is used for training the model
    #the mean squared error is used as the loss function which is typical for
    # regression tasks because it penalizes large errors more than small ones 
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

#------------------------------------------------------------------------------
# Prepare multistep data 
# this is used to create training data for a multistep time series forecasting model 
# it generates the input sequences(x) and the corresponding output sequences y form the given data 
#------------------------------------------------------------------------------

def prepare_multistep_data(data, sequence_length, k):

    # this x list will hold the input sequences and y list will hold the correspondign 
    # target values 
    X, y = [], []

    #this loop iterates over the dataset to generate input output pairs for training
    #i:i + sequence_length: For each iteration, the slice data[i:i + sequence_length] 
    # takes a sequence of sequence_length time steps.This sequence becomes the input 
    # (history) for the model. These values are appended to the X list.

    #i + sequence_length:i + sequence_length + k: After the sequence_length, 
    # the next k time steps (data[i + sequence_length:i + sequence_length + k]) 
    # are taken as the target values (what the model should predict). These values are 
    # appended to the y list.

    for i in range(len(data) - sequence_length - k + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + k])


        #After generating all the input-output pairs, both X and y lists are converted to NumPy arrays,
        #  which are the typical format for working with machine learning models 
        # (especially for deep learning).

    return np.array(X), np.array(y)

#------------------------------------------------------------------------------
# multistep prediction
# this function is used to predict future values using deep learning model
# the function takes in parameters like model, input data , sequence_length and K 
#------------------------------------------------------------------------------
def multistep_predict(model, input_data, sequence_length, k):
    print(f"Input data size: {input_data.size}")
    print(f"Expected size for reshaping: {sequence_length}")
    
    #this is to ensure that the input data has the expected number of elements if not it will raise an error
    if input_data.size != sequence_length:
        raise ValueError(f"Input data size {input_data.size} does not match the expected size {sequence_length}")
    
    # this line reshapes the input data into a 3D array with shape 1, sequence_length,1
    input_data = np.reshape(input_data, (1, sequence_length, 1))
    
    #this line is to make a prediction using the reshaped input data 
    prediction = model.predict(input_data)
    
    #this reshaped the prediction output to a 2D array with shape (k,1) where k is the number of 
    #predicted values
    prediction = prediction.reshape(-1, 1)

    return prediction


