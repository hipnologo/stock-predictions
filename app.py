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
    # This is just an example. You might need to update the URL or the scraping logic
    # as the Wikipedia page structure could change over time.
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = table['Symbol'].tolist()
    tickers = ['META' if ticker=='FB' else ticker for ticker in tickers]
    return tickers

def pull_momentum_stocks(tickers, momentum_threshold=0.05, json_file_path='tickers_momentum.json'):
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
    def fetch_stock_data(ticker):
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
        try:
            data.to_csv(f'data/{ticker}_stock_data.csv')
        except Exception as e:
            logging.error(f"Error saving data for ticker {ticker}: {str(e)}")

    def create_model(X_train):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model

    def train_and_predict_stock_price(ticker, data, look_ahead=60):
        try:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[['Close']].values)

            X_train, X_test, y_train, y_test = train_test_split(scaled_data[:-look_ahead], scaled_data[look_ahead:], shuffle=False)

            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            model = create_model(X_train)
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0, shuffle=False)

            predictions = model.predict(X_test[-look_ahead:])
            predictions = scaler.inverse_transform(predictions)

            # Save predictions with dates
            prediction_dates = data.index[-look_ahead:]
            predictions_df = pd.DataFrame(predictions, index=prediction_dates, columns=['Prediction'])
            predictions_df.to_csv(f'predictions/{ticker}_predictions.csv')

        except Exception as e:
            logging.error(f"Error in prediction for {ticker}: {str(e)}")


    #tickers = load_tickers_from_json('tickers.json')
    
    for ticker in tickers:
        data = fetch_stock_data(ticker)
        if data is not None:
            save_stock_data_to_csv(ticker, data)
            train_and_predict_stock_price(ticker, data)

def visualize_charts_and_predictions(tickers):
    def plot_predictions(ticker):
        try:
            predictions = pd.read_csv(f'predictions/{ticker}_predictions.csv', index_col=0)
            predictions.index = pd.to_datetime(predictions.index)
            plt.figure(figsize=(14, 8))

            # Calculate moving averages
            predictions['MA21'] = predictions['Prediction'].rolling(window=21).mean()
            predictions['MA200'] = predictions['Prediction'].rolling(window=200).mean()

            plt.plot(predictions['Prediction'], label='Prediction')
            plt.plot(predictions['MA21'], label='21 periods Moving Average', color='orange')
            plt.plot(predictions['MA200'], label='200 periods Moving Average', color='green')

            plt.title(f'Predictions and Moving Averages for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            
            # Rotate date labels
            plt.xticks(rotation=90)

            plt.show()
        except FileNotFoundError:
            print(f"No prediction data found for {ticker}")

    #tickers = load_tickers_from_json('tickers.json')
    
    for ticker in tickers:
        plot_predictions(ticker)

def pull_sentiment_analysis_for_stocks(tickers):
    def analyze_sentiment(tickers):
        # Getting Finviz Data
        news_tables = {}        # contains each ticker headlines
        for ticker in tickers:
            url = f'https://finviz.com/quote.ashx?t={ticker}'
            req = Request(url=url, headers={'user-agent': 'news'})
            response = urlopen(req)     # taking out html response
                    
            html = BeautifulSoup(response, features = 'html.parser')
            news_table = html.find(id = 'news-table') # gets the html object of entire table
            news_tables[ticker] = news_table

        ignore_source = ['Motley Fool', 'TheStreet.com'] # sources to exclude

        # Parsing and Manipulating
        parsed = []    
        for ticker, news_table in news_tables.items():  # iterating thru key and value
            for row in news_table.findAll('tr'):  # for each row that contains 'tr'
                if row.a is not None:    # Check if 'a' tag exists in the row
                    title = row.a.text
                    source = row.span.text
                    date_data = row.td.text.split(' ')
                    if len(date_data) > 1:     # both date and time, ex: Dec-27-20 10:00PM
                        date = date_data[0]
                        time = date_data[1]
                    else:
                        time = date_data[0] # only time is given ex: 05:00AM

                    if source.strip() not in ignore_source:
                        parsed.append([ticker, date, time, title])                              

        # Applying Sentiment Analysis
        df = pd.DataFrame(parsed, columns=['Ticker', 'date', 'Time', 'Title'])
        vader = SentimentIntensityAnalyzer()

        # for every title in data set, give the compound score
        score = lambda title: vader.polarity_scores(title)['compound']
        df['compound'] = df['Title'].apply(score)   # adds compound score to data frame

        # Visualization of Sentiment Analysis
        df['date'] = pd.to_datetime(df.date).dt.date # takes date column convert it to date/time format

        plt.figure(figsize=(6,6))      # figure size
        # unstack() allows us to have dates as x-axis
        mean_df = df.groupby(['date', 'Ticker']).mean() # avg compound score for each date
        mean_df = mean_df.unstack() 

        # xs (cross section of compound) get rids of compound label
        mean_df = mean_df.xs('compound', axis="columns")
        mean_df.plot(kind='bar')
        plt.show()

    #tickers = load_tickers_from_json('tickers_momentum.json')

    analyze_sentiment(tickers)

def main():
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
                pull_sentiment_analysis_for_stocks(tickers)

        elif option == '5':
            break

        else:
            print("Invalid option selected. Please try again.")

if __name__ == "__main__":
    main()