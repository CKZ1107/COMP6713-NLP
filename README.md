# Stock Prediction

## 1. Dataset
We will be using data that directly relates to a specific stock, as well as what we refer to as _external_ data, which may influence every stock. We use Twitter tweets from the [Stock Tweets for Sentiment Analysis and Prediction](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction?resource=download&select=stock_tweets.csv) dataset to simulate direct influence - these are tweets that directly reference the stock in question, and Reddit posts from [Daily Financial News for 6000+ Stocks](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?select=raw_partner_headlines.csv) to simulate the external influences.

In our combined dataset, we refer to these external-influence entries as stock _N/A_, which is simply a special keyword to imply that these posts are to be considered for all stocks.

## 2. Saved Fine-tuned model
https://unsw.sharepoint.com/:f:/r/sites/COMP6713977/Shared%20Documents/Chat/saved_HBERT_model?csf=1&web=1&e=YO7hDW
