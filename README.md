# Deep Learning Algorithmic Trading Bot
This project features an algorithmic swing trading bot that utilizes a recurrent neural network (LSTM) to classify stock returns. The bot is engineered to execute trades based on these predicted classifications, offering an empirical approach to swing trading. Additionally, it integrates insider trading data to enhance trading decisions by tracking company executives' significant buy/sell activities. 

## Initial Considerations
Embarking on this ambitious project involves addressing several key factors. First and foremost, knowing how to work with time series data is crucial, as stock prices are inherently correlated with time. Forecasting stock prices thus reveals itself as a sequence-to-sequence (seq-2-seq) problem, necessitating extensive data transformation to ensure it is suitable for our model.

Another significant consideration is the high dimensionality of the data. This project utilizes data from 464 stocks within the S&P 500, tracking their historical movements from January 2014 to August 2024, amounting to approximately 1.2 million observations. To balance data relevancy and computational efficiency, we must employ various techniques, such as standardization, categorical encoding, and dimensionality reduction.

Additionally, ensuring the quality and consistency of the data is paramount. We need to handle missing values, outliers, and any potential anomalies to maintain the integrity of the dataset. Considering the potential for class imbalance in stock return classifications, techniques such as class weighting or resampling may be necessary to ensure robust model training.

Lastly, developing a robust entry trading strategy necessitates crafting an entry and exit strategy capable of adapting to a volatile and stochastic market. This involves optimizing portfolio allocation, setting precise stop losses, and defining clear criteria for when to buy and sell stocks. The goal is to ensure that the trading strategy remains effective and resilient under varying market conditions.

## Implementing the LSTM Model
### Why LSTM?
Neural networks differ from traditional machine learning classifiers, such as SVMs or Decision Trees, in their ability to improve performance as the dataset size increases. This improvement is due to the vast number of tunable parameters in neural networks, which can range from thousands to even trillions. These parameters enable neural networks to generalize complex patterns and phenomena, effectively learning from a large number of scenarios through extensive training.

A specific subcategory of neural networks particularly adept at handling seq-2-seq problems is the recurrent neural network (RNN). Unlike traditional neural networks, which process each input and output independently, RNNs retain information from previous inputs to inform subsequent outputs. This capability allows RNNs to effectively capture and utilize the sequential context of the data. The Long Short-Term Memory (LSTM) model, a variant of the RNN, has a unique architecture that captures long-range dependencies better than a basic RNN. Given that our dataset spans over a decade, the LSTM model is the most suitable choice for this task.

