import numpy as np
import matplotlib.pyplot as plt



#------------------------------------------------------------------------------
# plot Boxplot chart function
#------------------------------------------------------------------------------
#this box plot function has 3 parameters which is DF, column and window
#the DF is for data frame that contains the stock market data
#The column is to specify which column to be analyzed 
#the window is the number of trading days to include in each window 
def plot_stock_boxplot(df, column='Close', window=5):
    
    #this if statement is to check if the column exists in the data frame 
    # if it is not it will raise the error
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    #The num_windows will calculate the number of windows that can be created based on the window size
    #and length of the data set if the size is to large for the dataset it will raise an error
    num_windows = len(df) - window + 1
    if num_windows <= 0:
        raise ValueError(f"Window size {window} is too large for the dataset length.")

    #data_for_boxplot is a list that will hold that data for each window
    # the loop iterates through the dataframe slicing the data into windows of size and append each slice
    #into the data_for_boxplot
    data_for_boxplot = []
    for i in range(num_windows):
        window_data = df[column].iloc[i:i + window].values
        data_for_boxplot.append(window_data)

    #the plt.figure(figsize=(10,6)) is used to set the figure size 
    #the plt.boxplot(data_for_boxplot, showfliers=False) is used to create the boxplot without showing outliers
    #the plt.title is used to set the title of the plot
    #the plt.xlable and plt.ylable is used to set the lables for the x and y axis
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_for_boxplot, showfliers=False)
    plt.title(f'Boxplot of {column} over a Moving Window of {window} Days')
    plt.xlabel(f'Moving Windows ({window} Days Each)')
    plt.ylabel(f'{column} Value')

    #plt.xticks is used to set the x axis tick lable and it creates tick at intervals that is 
    #calculated by max(1, num_window //10) and lables them with the range of days in each window
    plt.xticks(np.arange(1, num_windows + 1, max(1, num_windows // 10)),
               [f'{i+1}-{i+window}' for i in range(0, num_windows, max(1, num_windows // 10))])
    
    #plt.grid(true) is used to add a grid to the plot
    #plt.show is used to display the plots
    plt.grid(True)
    plt.show()