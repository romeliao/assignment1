from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,GRU,Bidirectional,InputLayer



#------------------------------------------------------------------------------
# create Deep learning model function
#------------------------------------------------------------------------------
#this functions takes multiple parameters. 
#The sequence length is used to represent the number of time steps in each input sequence 
#The n_features is used for the number of features in each time step
#The units are used for the number of units in each LSTM cell which by default is 256
#The cell is the type of deep learning network that is being used 
#n_layers is the number of deep learning layers that is in the model 
#The dropout is used to prevent overfitting by randomly setting a fraction of the input units to 0 during training 
#loss is used calculate the difference between the predicted outputs and the actual target values in the dataset 
#the optimizer is an algorithm that updates the model's weights to minimize the loss function.
#bidirectional is a boolean that indicates whether to use the Bidirectional layers.
def create_model(sequence_length, n_features, units=256, cell=GRU,n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    
    #create model
    #this is to initialize the model and the sequential model is a linear stack of layers 
    #each layer has one input tensor and one output tensor.
    model = Sequential()

    #for loop where n_layers defnes how many layers to be added 
    for i in range(n_layers):
        
        
        if i == 0:
            #first layer
            #if bidirectional is true there will be a bidirectional layer added. 
            #the layer will process the input sequence in both forward and backward direction and the output 
            #will be a sequence of the same length as the input
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), 
                                        batch_input_shape=(None, sequence_length, n_features))) 
            else: 
                #if bidirectional is false a undirectional RNN layer is added.
                #the layer will process the input sequence in only one direction which is forward 
                #and the output will be a sequence of the same length as the input 
                model.add(cell(units, return_sequences=True, 
                               batch_input_shape=(None, sequence_length, n_features)))
        
        elif i == n_layers - 1:
            #last layer 
            #if bidirectional is true there will be a bidirectional layer added
            #this layer will process the input sequence in both forward and backward directions 
            #but the output will be a fixed size vector and not a sequence 
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                #if its false a undirectional layer is added. this means the layer process
                #the input sequecne in only one direction and the output will be a fixed size vector
                model.add(cell(units, return_sequences=False))
        else:
            #hidden layers
            #if bidirectional is true a bidirectional layer is added with the 
            #return_sequences = true this mean that the layer will process the input sequence in 
            #both forward and backward directions, the output wll be a sequence of the smae length as the input 
            if bidirectional:
                model.add(Bidirectional(cell(units,return_sequences=True)))
            else:
            #if its false a undirectional layer is added with teh return sequence True.
            #this is teh layer taht process the input sequence in only one direcional and the output 
            #will be a sequence of the same length as the input 
                model.add(cell(units, return_sequences=True))

    #add dropout after each layer to apply dropout reularization
    #this line addes a dropout layer to the model where dropout is a float value between 0 and 1
    #that represents teh fraction of neurns to randomly drop during training 
    model.add(Dropout(dropout))

    #this line is to add a final dense layer to the model with a single output neuron 
    #and a linear activation function 
    #the output of this layer will be a single value which is suitable for regression 
    #problems.
    model.add(Dense(1, activation="linear"))

    #this line compiles the model specifying the loss function which is typically mean squared error or mean absolute error 
    #for regression problems 
    #the metrics=["mean_absolute_error"] specifies that the model should track the mean absoute error during training 
    #in addition to the loss function
    # the optimization algorithm is used during training.
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model