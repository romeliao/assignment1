import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM



#------------------------------------------------------------------------------
# multivariate_prediction function 
#------------------------------------------------------------------------------
# this function takes in parameter that takes the company name of whose stock data to predict 
# it also takes the start_date and end date to filter the stock data 
# the prediction_days is used for the previous days to use for making a prediction 
# features is used to get the list of features to be used for the model
def multivariate_prediction(company, start_date, end_date, prediction_days, features):
    
    #this is to select which features to predict
    # list of features ['Open','High','Low','Close','Adj Close','Volume']
    selected_feature = 'Volume'

    # Load data
    # this is to call the dateset into the model
    data = pd.read_csv(f"csv-results/CBA.AX_2024-09-14.csv")

    # Convert date to datetime and set as index
    # this is to convert the date to a datetime object for easy filtering 
    # the date column is sed to set as the index for the data Dataframe to allow time based operations 
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Filter data by date
    # this filtered the data to only nclude the rows between the start and end date
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # Scale data
    # the function uses a MinMaxScaler to scale the data between 0 and 1 for all the selected features
    # this will help thw model convergence during the training since neural networks perform better 
    # when values are normalized 
    # FEATURES only the specidied features like ['Open','High','Low','Close','Adj Close','Volume'] are selected
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[features])

    # Prepare data
    x = []
    y = []

    # input x is for each day in the dataset the function collects the previous prediction_days 
    #of fata across all features.
    #  the output y for each corresponding input it collects the scaled selected function price for the next day
    for i in range(prediction_days, len(scaled_data)):
        x.append(scaled_data[i-prediction_days:i, :])
        y.append(scaled_data[i, features.index(selected_feature)])  # Assuming 'Close' is one of the features

    #this is to convert the x and y list are converted to NumPy arrays which is the expected format for 
    #Training the deep learning model
    x = np.array(x)
    y = np.array(y)

    # Split data into training and testing sets
    # this is where the data is split into training and testing sets with 80% for traning and the remaining for testing 
    # thsi split ensures the model can be evaluated on unseen data
    train_size = int(0.8 * len(x))
    x_train, x_test = x[0:train_size], x[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]

    # Reshape data for LSTM
    # the x_train and x__test arrays are reshaped to te required 3D shape for LSTM models
    # x_train.shape[0]: Number of samples.
    # x_train.shape[1]: Number of time steps (equal to prediction_days).
    # len(features): The number of features per time step (e.g., 'Open', 'Close', etc.).
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features)))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(features)))

    # Build model
    # the first LSTM layer returns sequences
    # the second LSTM layer returns a signle output
    # the dropout lyaers are added to reduce overfitting by randomly setting some of the layer outputs 
    # to zero during training 
    # the final Dense layer outputs a single value which represents the predicted 
    # selected value price for the next day 
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile model
    # the adam optimizer is used for effcient training 
    # the model uses mean squared error as the loss function which is suitable for regression problems
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Make prediction
    # the trainind model makes predictions on the test set which are scaled values of the selected 
    # function price 
    prediction = model.predict(x_test)

    # Reshape the prediction to match the feature space used for scaling
    # the predicted values are placed into a matrix that matches the shape of the scaled data filling in only
    # the selected features
    prediction_reshaped = np.zeros((prediction.shape[0], len(features)))
    prediction_reshaped[:, features.index(selected_feature)] = prediction.flatten()

    # Inverse transform the prediction
    # the scaled predictions are transformed back to their original scale using the scaler. inverse_transform
    # this returns the actual features prices whcih can be compared with the real test values.
    prediction = scaler.inverse_transform(prediction_reshaped)[:, features.index(selected_feature)]

    print(f"\n Multivariate_predictions of {selected_feature} function: \n {prediction}")

    return prediction