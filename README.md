# Deep Learning Algorithmic Trading Bot
This project features an algorithmic swing trading bot that utilizes a recurrent neural network (LSTM) to classify stock returns. The bot is engineered to execute trades based on these predicted classifications, offering an empirical approach to swing trading. Additionally, it integrates insider trading data to enhance trading decisions by tracking company executives' significant buy/sell activities. 

## Initial Considerations
Embarking on this ambitious project involves addressing several key factors. First and foremost, working with time series data is crucial, as stock prices are inherently correlated with time. Forecasting stock prices is thus a sequence-to-sequence (seq-2-seq) problem, necessitating extensive data transformation to ensure it is suitable for our model.

Another significant consideration is the high dimensionality of the data. This project utilizes data from 464 stocks within the S&P 500, tracking their historical movements from January 2014 to August 2024, amounting to approximately 1.2 million observations. To balance data relevancy and computational efficiency, we must employ various techniques, such as standardization, dimensionality reduction, and feature selection.

Additionally, ensuring the quality and consistency of the data is paramount. We need to handle missing values, outliers, and any potential anomalies to maintain the integrity of the dataset. Considering the potential for class imbalance in stock return classifications, techniques such as class weighting or resampling may be necessary to ensure robust model training.

## Implementing the LSTM Model
