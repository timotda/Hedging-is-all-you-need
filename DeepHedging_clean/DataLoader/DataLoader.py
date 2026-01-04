import yfinance as yf
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class DataLoader:
    """
    This class should be used for downloading and loading stock market data using yfinance.
    This is a mandatory step before runing main.py !

    
    Attributes:
        tickers (list): List of stock ticker symbols to download/load
        start_date (str): Start date for historical data (format: 'YYYY-MM-DD')
        end_date (str): End date for historical data (format: 'YYYY-MM-DD')
        directory_path (str): Directory path where CSV files will be saved/loaded from
    """
    
    def __init__(self, tickers, start_date, end_date, directory_path='DeepHedging_clean/data'):
        """        
        Args:
            tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            directory_path (str, optional): Path to directory for saving/loading data.
                Defaults to 'DeepHedging_clean/data'.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.directory_path = directory_path
    
    def download_data(self):
        """        
        Downloads data for each ticker in self.tickers for the specified date range
        and saves each ticker's data to a separate CSV file in the directory_path.
        
        Returns:
            None. Files are saved to disk at {directory_path}/{ticker}.csv
        """
        # create or check that the output directory exists 
        Path(self.directory_path).mkdir(parents=True, exist_ok=True)
        print(f"Begin Download...")
        
        for ticker in tqdm(self.tickers):
            data = yf.download(ticker, self.start_date, self.end_date)['Close']
            data.to_csv(Path(self.directory_path) / f"{ticker}.csv")
            print(f"Data for {ticker} has been downloaded and saved to {self.directory_path}/{ticker}.csv")

    def load_data(self, tickers):
        """        
        Loads closing price data for a specified list of tickers from CSV files
        and combines them into a single DataFrame with Date as index and one column
        per ticker.
        
        Args:
            tickers (list): List of ticker symbols to load. Can be a subset of
                self.tickers to load only specific stocks.
        
        Returns:
            pd.DataFrame: DataFrame with Date index and columns for each ticker
                containing closing prices.
        """
        # here tickers can be used to load only a specific list of tickers, not all the ones
        print(F"Loading data")
        Path(self.directory_path).mkdir(parents=True, exist_ok=True)

        data = pd.DataFrame()
        for ticker in tqdm(tickers):
            data[ticker] = pd.read_csv(Path(self.directory_path) / f"{ticker}.csv",parse_dates=["Date"]).set_index("Date")
        return data


if __name__ == '__main__':
    
    # Change values here to download the data you want
    tickers = ['AAPL','GOOGL','MSFT','AMZN','BRK-B']
    start_date = '2008-09-26'   
    end_date = '2025-01-01'
    data_loader_object = DataLoader(tickers, start_date, end_date)
    data_loader_object.download_data()
    data = data_loader_object.load_data(['AAPL','GOOGL','MSFT','AMZN','BRK-B'])
    data.to_csv("../Data/stocks_close_prices_2008_2025.csv")
    print(data)
