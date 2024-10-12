from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# ARIMA Model Function
# The function takes in parameters liek train_data and order 
# Train_data is the time seriese data that the ARIMA model will be trained on
# the order is the tuple that defines the paremeters of the ARIMA model where 
# p is the number of lag observations 
# d is the number of times the data needs to be differenced to make t stationary 
# q is the size of the movin agerage window 
#------------------------------------------------------------------------------
# the tuple that defines the parameter of the ARIMA model 
ARIMA_ORDER = (15, 2, 0)           


def train_arima_model(train_data, order=ARIMA_ORDER):
    try:
        # Create the ARIMA model using the training data and specified order
        model = ARIMA(train_data, order=order)
        
        # Fit the ARIMA model to the training data and estimate the coefficients
        model_fit = model.fit()
        
        # Return the fitted model
        return model_fit
    
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None
    

def plot_predictions(actual_prices, arima_predictions, predicted_prices, ensemble_predictions, company_name):
    """
    Plots the actual prices and predictions from ARIMA, deep learning (DL), and ensemble models.
    
    Parameters:
    - actual_prices: array-like, the actual stock prices.
    - arima_predictions: array-like, predictions from the ARIMA model.
    - predicted_prices: array-like, predictions from the deep learning model.
    - ensemble_predictions: array-like, predictions from the ensemble model.
    - company_name: str, the name of the company whose stock prices are being plotted.
    """
    # Create a plot with a defined figure size
    plt.figure(figsize=(14, 7))

    # Plot actual prices
    plt.plot(actual_prices, color="black", label=f"Actual {company_name} Price")

    # Plot ARIMA predictions
    plt.plot(arima_predictions, color="red", label="ARIMA Predictions")

    # Plot deep learning model predictions
    plt.plot(predicted_prices, color="green", label="DL Predictions")

    # Plot ensemble model predictions
    plt.plot(ensemble_predictions, color="blue", label="Ensemble Predictions")

    # Add plot title and axis labels
    plt.title(f"{company_name} Share Price Predictions")
    plt.xlabel("Time")
    plt.ylabel(f"{company_name} Share Price")

    # Add a legend to differentiate the different lines
    plt.legend()

    # Display the plot
    plt.show()

def print_arima_predictions(arima_predictions):
    #this is used to print out the predicted values form the ARIMA model
    print(f"\nARIMA Predictions:\n{arima_predictions}")