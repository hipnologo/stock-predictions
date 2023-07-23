# Stock Predictions

This is a Python application that uses various libraries and techniques to analyze and predict stock performance. The application pulls stock data, analyzes momentum, runs price predictions using Machine Learning, and performs sentiment analysis on news related to the stocks. 

## Features

1. **Pull momentum stocks**: The application can identify stocks with strong momentum, defined by a significant change in price over a certain period.
2. **Run price predictions**: The application uses a LSTM (Long Short-Term Memory) model to predict future stock prices based on historical data.
3. **Visualize prediction charts**: The application can generate charts showing the actual stock prices, moving averages, and predicted prices.
4. **Pull sentiment analysis for the stocks**: The application performs sentiment analysis on news related to the stocks, which can provide insights about public sentiment towards the companies.

## Prerequisites

- Python 3.7+
- pip

## Installation

1. Clone this repository: `git clone https://github.com/hipnologo/stock-predictions.git` (Replace "username" and "repo" with your GitHub username and repository name)
2. Navigate to the cloned directory: `cd repo`
3. Install the requirements: `pip install -r requirements.txt`

## Usage

Run `python main.py` and follow the prompts.

## Libraries Used

- pandas
- yfinance
- json
- numpy
- sklearn
- tensorflow
- matplotlib
- nltk
- BeautifulSoup

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute and the process for submitting pull requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

If you have any questions, feel free to reach out to us.
