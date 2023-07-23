import pandas as pd
import yfinance as yf
import json
import logging
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date

# Set up logging
logging.basicConfig(filename='stock_predictor.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

def load_tickers_from_json(json_file_path):
    """
    Loads the list of tickers from a JSON file.

    Args:
        json_file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        with open(json_file_path, 'r') as file:
            tickers = json.load(file)
    except FileNotFoundError:
        print(f"No tickers file found. Generating default tickers file.")
        # default to top 10 S&P500 stocks
        tickers = get_sp500_tickers()[:10]
        with open(json_file_path, 'w') as file:
            json.dump(tickers, file)
    return tickers

def get_sp500_tickers():
    """
    Pulls the list of S&P500 tickers from Wikipedia.

    Returns:
        _type_: _description_
    """    
    # Check if Wikipedia page structure could change over time.
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = table['Symbol'].tolist()
    tickers = ['META' if ticker=='FB' else ticker for ticker in tickers]
    return tickers

def pull_momentum_stocks(tickers, momentum_threshold=0.05, json_file_path='tickers_momentum.json'):
    """
    Pulls the list of momentum stocks and saves them to a JSON file.

    Args:
        tickers (_type_): _description_
        momentum_threshold (float, optional): _description_. Defaults to 0.05.
        json_file_path (str, optional): _description_. Defaults to 'tickers_momentum.json'.

    Returns:
        _type_: _description_
    """
    def calculate_momentum(price_data, period):
        return price_data['Close'].diff(period) / price_data['Close'].shift(period)

    def get_sp500_tickers():
        # This is just an example. You might need to update the URL or the scraping logic
        # as the Wikipedia page structure could change over time.
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()

    momentum_period = 7  # chosen momentum period
    #momentum_threshold = 0.07  # chosen momentum threshold

    tickers = get_sp500_tickers()
    momentum_tickers = []

    for ticker in tickers:
        try:
            # Yahoo Finance uses '-' instead of '.' in ticker symbols
            ticker = ticker.replace('.', '-')
            data = yf.download(ticker, period='1y')
            momentum = calculate_momentum(data, momentum_period)
            if momentum[-1] > momentum_threshold:  # if the most recent momentum score is above the threshold
                momentum_tickers.append(ticker)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {str(e)}")
            logging.error(f"Error processing ticker {ticker}: {str(e)}")

    # Save the tickers to a JSON file
    with open(json_file_path, 'w') as f:
        json.dump(momentum_tickers, f)
        print(f"{json_file_path} file created successfully!")

def run_predictions(tickers):
    """
    Predicts the stock prices for the given tickers and saves the predictions to a CSV file.

    Args:
        tickers (_type_): _description_
    """    
    def fetch_stock_data(ticker):
        """
        Fetches the stock data for the given ticker.

        Args:
            ticker (_type_): _description_

        Returns:
            _type_: _description_
        """        
        try:
            data = yf.Ticker(ticker).history(period='5y')
            if data.empty:
                logging.warning(f"No data fetched for ticker: {ticker}")
                return None
        except Exception as e:
            logging.error(f"Error fetching data for ticker {ticker}: {str(e)}")
            return None
        return data

    def save_stock_data_to_csv(ticker, data):
        """
        Saves the stock data to a CSV file.

        Args:
            ticker (_type_): _description_
            data (_type_): _description_
        """        
        try:
            data.to_csv(f'data/{ticker}_stock_data.csv')
        except Exception as e:
            logging.error(f"Error saving data for ticker {ticker}: {str(e)}")

    def create_model(X_train):
        """
        Creates the LSTM model.

        Args:
            X_train (_type_): _description_

        Returns:
            _type_: _description_
        """        
        model = Sequential()
        # Change the input_shape to match the number of timesteps (60 in your case)
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model

    def train_and_predict_stock_price(ticker, data, look_ahead=60, future_days=10):
        """
        Trains the model and predicts the stock price for the given ticker.

        Args:
            ticker (_type_): _description_
            data (_type_): _description_
            look_ahead (int, optional): _description_. Defaults to 60.
            future_days (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """        
        try:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[['Close']].values)

            X_train, X_test, y_train, y_test = train_test_split(scaled_data[:-look_ahead], scaled_data[look_ahead:], shuffle=False)

            def create_dataset(X, y, time_steps=1):
                Xs, ys = [], []
                for i in range(len(X) - time_steps):
                    v = X[i:(i + time_steps)]
                    Xs.append(v)
                    ys.append(y[i + time_steps])
                return np.array(Xs), np.array(ys)

            time_steps = 60

            # reshape to [samples, time_steps, n_features]
            X_train, y_train = create_dataset(X_train, y_train, time_steps)
            X_test, y_test = create_dataset(X_test, y_test, time_steps)

            model = create_model(X_train)
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0, shuffle=False)

            # Predict the future prices
            future_prices = []
            current_batch = scaled_data[-look_ahead:].reshape((1, look_ahead, 1))

            for i in range(future_days):
                # Ensure that current_batch has shape (1, 60, 1)
                future_price = model.predict(current_batch)[0]
                future_prices.append(future_price)
                current_batch = np.append(current_batch[:,1:,:], [[future_price]], axis=1)
                current_batch = current_batch.reshape(1, 60, 1)

            future_prices = scaler.inverse_transform(future_prices)

            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]

            # Save predictions with dates
            predictions_df = pd.DataFrame(future_prices, index=future_dates, columns=['Prediction'])
            predictions_df.to_csv(f'predictions/{ticker}_predictions.csv')

        except Exception as e:
            logging.error(f"Error in prediction for {ticker}: {str(e)}")

    #tickers = load_tickers_from_json('tickers.json')

    for ticker in tickers:
        print(f"Processing ticker {ticker}...")
        data = fetch_stock_data(ticker)
        if data is not None:
            print(f"Saving stock data for {ticker}...")
            save_stock_data_to_csv(ticker, data)
            print(f"Training model and predicting prices for {ticker}...")
            train_and_predict_stock_price(ticker, data)
            print(f"Predictions for {ticker} completed.")
        else:
            print(f"No data available for ticker {ticker}. Skipping...")
    print("All predictions completed.")
    
def visualize_charts_and_predictions(tickers):
    """
    Visualizes the charts and predictions for the given tickers.

    Args:
        tickers (_type_): _description_
    """    
    def select_data_range():
        print("\n1. Last 1 month")
        print("2. Last 3 months")
        print("3. Last 6 months")
        print("4. Last 1 year")
        print("5. Last 2 years")
        option = input("Select data range for the chart: ")

        if option == '1':
            return '1M'
        elif option == '2':
            return '3M'
        elif option == '3':
            return '6M'
        elif option == '4':
            return '1Y'
        elif option == '5':
            return '2Y'
        else:
            print("Invalid option. Defaulting to 1M.")
            return '1M'

    def filter_data_by_date_range(data, range_option):
        """
        Filters the data based on the selected range.

        Args:
            data (_type_): _description_
            range_option (_type_): _description_

        Returns:
            _type_: _description_
        """        
        if range_option.endswith('M'):
            months = int(range_option[:-1])
            return data.last(f'{months}M')
        elif range_option.endswith('Y'):
            years = int(range_option[:-1])
            return data.last(f'{years}Y')

    def plot_predictions(ticker, range_option):
        """
        Plots the predictions for the given ticker.

        Args:
            ticker (_type_): _description_
            range_option (_type_): _description_
        """        
        try:
            predictions = pd.read_csv(f'predictions/{ticker}_predictions.csv', index_col=0)
            predictions.index = pd.to_datetime(predictions.index)

            stock_data = pd.read_csv(f'data/{ticker}_stock_data.csv', index_col=0)
            stock_data.index = pd.to_datetime(stock_data.index)

            # Filter the data based on the selected range
            latest_date = max(stock_data.index.max(), predictions.index.max())
            if range_option.endswith('M'):
                months = int(range_option[:-1])
                earliest_date = latest_date - pd.DateOffset(months=months)
            elif range_option.endswith('Y'):
                years = int(range_option[:-1])
                earliest_date = latest_date - pd.DateOffset(years=years)

            stock_data = stock_data.loc[earliest_date:]
            predictions = predictions.loc[earliest_date:]

            plt.figure(figsize=(14, 8))

            # Plot the actual prices
            plt.plot(stock_data['Close'], label='Actual Price')

            # Calculate moving averages
            stock_data['MA21'] = stock_data['Close'].rolling(window=21).mean()
            stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

            plt.plot(stock_data['MA21'], label='21 periods Moving Average', color='orange')
            plt.plot(stock_data['MA200'], label='200 periods Moving Average', color='green')

            # Plot the predictions
            plt.plot(predictions['Prediction'], label='Predicted Price')

            plt.title(f'Actual Prices, Moving Averages, and Predictions for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Rotate date labels
            plt.xticks(rotation=90)

            plt.show()
        except FileNotFoundError:
            print(f"No prediction data found for {ticker}")

    data_range = select_data_range()

    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        print("Fetching prediction data...")
        plot_predictions(ticker, data_range)
        print(f"Finished processing {ticker}\n")

def pull_sentiment_analysis_for_stocks(tickers, source='finviz'):
    """
    Pulls the sentiment analysis for the given tickers and source.

    Args:
        tickers (_type_): _description_
        source (str, optional): _description_. Defaults to 'finviz'.
    """    
    def analyze_sentiment(tickers, source):
        """
        Analyzes the sentiment for the given tickers and source.

        Args:
            tickers (_type_): _description_
            source (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # Different parsing functions for different sources
        def parse_finviz(html):
            """
            Parses the Finviz HTML.

            Args:
                html (_type_): _description_

            Returns:
                _type_: _description_
            """            
            news_table = html.find('table', {'id': 'news-table'})
            return news_table

        def parse_yahoo(html):
            """
            Parse the Yahoo Finance HTML.

            Args:
                html (_type_): _description_

            Returns:
                _type_: _description_
            """            
            news_table = html.find('div', class_='news-container')  # Adjust this to match the actual tag and class name
            return news_table

        # Mapping of sources to parsing functions
        parse_functions = {
            'finviz': parse_finviz,
            'yahoo': parse_yahoo
        }

        # Choose the right parsing function
        parse_function = parse_functions[source]

        # Getting Data
        ignore_source = ['Motley Fool', 'TheStreet.com'] # sources to exclude
        news_tables = {}        
        for ticker in tickers:
            if source == 'finviz':
                url = f'https://finviz.com/quote.ashx?t={ticker}'
            elif source == 'yahoo':
                url = f'https://finance.yahoo.com/quote/{ticker}/news?p={ticker}'

            req = Request(url=url, headers={'user-agent': 'news'})
            response = urlopen(req)     
            html = BeautifulSoup(response, features='html.parser')

            # Call the chosen parsing function
            news_table = parse_function(html)
            news_tables[ticker] = news_table

        # Parsing and Manipulating
        parsed = []    
        for ticker, news_table in news_tables.items():  # iterating thru key and value
            if news_table is not None:
                news_items = news_table.find_all('div')  # Adjust this to match the actual tag of the individual news articles
                for news_item in news_items:
                    title = news_item.a.text if news_item.a else ''
                    source = news_item.span.text
                    date_data = news_item.td.text.split(' ')  # Corrected here
                    if len(date_data) > 1:     # both date and time, ex: Dec-27-20 10:00PM
                        date = date_data[0]
                        time = date_data[1]
                    else:
                        time = date_data[0] # only time is given ex: 05:00AM

                    if source.strip() not in ignore_source and date:
                        parsed.append([ticker, date, time, title]) 
            else:
                print(f"No news found for ticker {ticker} from source {source}")

        # Applying Sentiment Analysis
        df = pd.DataFrame(parsed, columns=['Ticker', 'date', 'Time', 'Title'])
        vader = SentimentIntensityAnalyzer()

        # for every title in data set, give the compound score
        score = lambda title: vader.polarity_scores(title)['compound']
        df['compound'] = df['Title'].apply(score)   # adds compound score to data frame

        # Visualization of Sentiment Analysis
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df.date, errors='coerce').dt.date # takes date column convert it to date/time format

            plt.figure(figsize=(6,6))      # figure size
            # unstack() allows us to have dates as x-axis
            mean_df = df.groupby(['date', 'Ticker']).mean() # avg compound score for each date
            mean_df = mean_df.unstack() 

            # xs (cross section of compound) get rids of compound label
            if 'compound' in mean_df.columns:  # Check if 'compound' is in columns before plotting
                mean_df = mean_df.xs('compound', axis="columns")
                mean_df.plot(kind='bar')
                plt.show()
            else:
                print("No sentiment analysis results to plot.")

    analyze_sentiment(tickers, source)

def main():
    """
    Main function to run the stock predictor app.
    """    
    while True:
        print("\n1. Pull momentum stocks")
        print("2. Run predictions")
        print("3. Visualize the charts and predictions")
        print("4. Pull sentiment analysis for the stocks")
        print("5. Exit")
        option = input("Select an option: ")

        if option == '1':
            tickers = load_tickers_from_json('tickers.json')
            print("\n1. Use default momentum threshold")
            print("2. Define momentum threshold")
            threshold_option = input("Select an option: ")
            if threshold_option == '1':
                pull_momentum_stocks(tickers, momentum_threshold=0.05)
            elif threshold_option == '2':
                custom_threshold = float(input("Enter your custom momentum threshold: "))
                pull_momentum_stocks(tickers, momentum_threshold=custom_threshold)
            else:
                print("Invalid option selected. Please try again.")

        elif option in ['2', '3', '4']:
            print("\n1. Use tickers.json")
            print("2. Use tickers_momentum.json")
            ticker_option = input("Select an option: ")
            if ticker_option == '1':
                tickers = load_tickers_from_json('tickers.json')
            elif ticker_option == '2':
                tickers = load_tickers_from_json('tickers_momentum.json')
            else:
                print("Invalid option selected. Please try again.")
                continue

            if option == '2':
                run_predictions(tickers)
            elif option == '3':
                visualize_charts_and_predictions(tickers)
            elif option == '4':
                print("\n1. Finviz")
                print("2. Yahoo")
                source = input("Select a News Source: ")
                if ticker_option == '1':
                    pull_sentiment_analysis_for_stocks(tickers, source='finviz')
                elif ticker_option == '2':
                    pull_sentiment_analysis_for_stocks(tickers, source='yahoo')
                else:
                    print("Invalid option selected. Please try again.")
                    continue

        elif option == '5':
            break

        else:
            print("Invalid option selected. Please try again.")

if __name__ == "__main__":
    main()