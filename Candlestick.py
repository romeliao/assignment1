import yfinance as yf
import mplfinance as mpf




#------------------------------------------------------------------------------
# plot candlestick chart function
#------------------------------------------------------------------------------
#this function has 4 parameters which is company, start and end date, and n_days
#the company parameter is the ticker symbol of the company 
#The start date and end date parameter is used to get the data within that range 
#the n_days is the numbers of days to resample the data
def plot_candlestick_chart(company, start_date, end_date,n_days=1):
    
    # this is to retrieve the historical stock data for the specified company and data range
    data = yf.download(company, start=start_date, end=end_date)

    #this if statement is to ensure that if n_days is greater than 1 the function will check the retrieved 
    #dataframe that contains all the required columns (open,high,low,close,volume) if there is none the 
    # will be a error that will be raised 
    if n_days > 1:
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("The DataFrame does not contain all required columns.")
        
        #this is used to resample the data to the specified number of days using the resample 
        #and aggregates teh values using the agg method
        data = data.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    #this is used to remove any rows with missing values 
    data.dropna(inplace=True)

    #plot the candlestick chart with the specified style, title and labels
    mpf.plot(data, type='candle', style='charles', title=f'{company} Candlestick Chart',
             ylabel='Price', volume=True)