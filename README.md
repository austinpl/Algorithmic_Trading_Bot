# Deep Learning Algorithmic Trading Bot
This project features an algorithmic swing trading bot that utilizes a recurrent neural network (LSTM) to classify stock returns. The bot is engineered to execute trades based on these predicted classifications, offering an empirical approach to swing trading. Additionally, it integrates insider trading data to enhance trading decisions by tracking company executives' significant buy/sell activities. 

## Initial Considerations
First and foremost, knowing how to work with time series data is crucial, as stock prices are inherently correlated with time. Forecasting stock prices thus reveals itself as a sequence-to-sequence (seq-2-seq) problem, necessitating extensive data transformation to ensure it is suitable for our model.

Another significant consideration is the high dimensionality of the data. This project utilizes data from 464 stocks within the S&P 500, tracking their historical movements from January 2014 to August 2024, amounting to approximately 1.2 million observations. To balance data relevancy and computational efficiency, we must employ various techniques, such as standardization, categorical encoding, and dimensionality reduction.

Additionally, ensuring the quality and consistency of the data is paramount. We need to handle missing values, outliers, and any potential anomalies to maintain the integrity of the dataset. Considering the potential for class imbalance in stock return classifications, techniques such as class weighting or resampling may be necessary to ensure robust model training.

Lastly, developing a robust entry trading strategy necessitates crafting an entry and exit strategy capable of adapting to a volatile and stochastic market. This involves optimizing portfolio allocation, setting precise stop losses, and defining clear criteria for when to buy and sell stocks. The goal is to ensure that the trading strategy remains effective and resilient under varying market conditions.

## Implementing the LSTM Model
### Why LSTM?
Neural networks differ from traditional machine learning classifiers, such as SVMs or Decision Trees, in their ability to improve performance as the dataset size increases. This improvement is due to the vast number of tunable parameters in neural networks, which can range from thousands to even trillions. These parameters enable neural networks to generalize complex patterns and phenomena, effectively learning from a large number of scenarios through extensive training.

A specific subcategory of neural networks particularly adept at handling seq-2-seq problems is the recurrent neural network (RNN). Unlike traditional neural networks, which process each input and output independently, RNNs retain information from previous inputs to inform subsequent outputs. This capability allows RNNs to effectively capture and utilize the sequential context of the data. The Long Short-Term Memory (LSTM) model, a variant of the RNN, has a unique architecture that captures long-range dependencies better than a basic RNN. Given that our dataset spans over a decade, the LSTM model is the most suitable choice for this task.

### Feature Engineering
Having chosen our model, the next step is to determine which factors (features) can help traders predict a stock's behavior. Stock trading heavily relies on technical analysis, using various indicators or metrics to gain a deeper understanding of price movements and chart trends. In addition to the standard stock information such as open, close, adjusted close, high, low, and volume, we will incorporate technical indicators like Garman-Klass volatility, RSI, MACD, EMA-10, and EMA-50. These indicators will provide our model with a more comprehensive understanding of the stock's behavior, enhancing its predictive accuracy. 

### Defining Model Output
Now begs the question: what exactly do we want the model to predict?

There are several approaches to this. One option is to have the model predict the exact price movement of a stock, which is a regression problem. Another method is to determine whether the stock will end positive or negative for the day, a classification problem. Given that stock movement is largely stochastic, I decided that a classification approach would provide the model with some flexibility, rather than requiring it to predict specific prices.

However, instead of binary classification, I opted for multi-class classification to capture the magnitude of a stock's price change. After all, a one-cent increase in stock price is very different from a ten-dollar increase, even though both outcomes are positive. To achieve this, I created four categories: greater profit, lesser profit, lesser losses, and greater losses. The distinction between "greater" and "lesser" is based on whether the stock ended within the top 50% or bottom 50% of their respective classification percentile.

In addition to capturing the magnitude of price changes, multi-class classification also provides the model with a greater margin for error. In this approach, perfect accuracy isn't required. For example, if the model predicts a stock will yield "greater profit" but it only results in "lesser profit," we still achieve a positive outcome.

### Data Transformation

### Model Creation and Training
