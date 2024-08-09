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
#### Principal Component Analysis (PCA)
A frequently overlooked issue in modeling is multicollinearity, where multiple features are highly correlated with each other. This can create redundancies within the model and diminish the true individual significance of the independent variables.

A powerful technique to address this is principal component analysis (PCA). PCA uses linear algebra (specifically eigenvectors and eigenvalues) to condense the information from the original features into a smaller set of new features, or principal components, while minimizing information loss. In our case, we transformed our features into 7 principal components, reducing the total number of features by 5. This approach allowed us to retain 99% of the explained variance, which represents an excellent trade-off between dimensionality reduction and information preservation.

#### Binary Encoding of Categorical Variables
Since we are dealing with multiple stocks, each with its own unique characteristics, it’s important to explicitly differentiate them so our model can learn the distinctions between each stock. To represent the identity of a stock in a way that our model can understand, we use a technique called categorical encoding.

One common method is one-hot encoding, which assigns a unique binary vector to each stock (e.g., 464 features, one for each stock). However, this would add 464 new features, which could overly complicate our model. Instead, I chose to use binary encoding, which represents each stock with a binary code, significantly reducing the number of new features to just 9. This keeps the model simpler and more efficient while still conveying the necessary information.

#### Sequence Creation for LSTM Input
With our data now effectively transformed to include the necessary information for the model, the final step is to format it for processing by the LSTM.

LSTMs have specific requirements for their inputs, as they process sequential data by breaking it into smaller segments called sequences. The context gathered from each sequence is then used to inform the output for the following step. Although our data is already in chronological order, we still need to create these sequences for each stock, covering the entire period from the starting date to the end date.

We can define the length of each sequence, known as the lookback period, which determines the timeframe the LSTM will consider as context for its next output. For this model, we'll use a lookback period of 30 days, meaning the model will use 30 days of prior data to inform its next prediction.

### Model Creation and Training
#### Dealing with More than One Observation per Timestamp
One of the biggest challenges I faced was adapting the LSTM to analyze data from 464 stocks simultaneously, given that LSTMs are typically designed to handle individual sequences. Since each stock shares the same timestamps, organizing the data for sequential processing by the LSTM was particularly challenging. Grouping the data by date would cause the LSTM to mistakenly interpret the next stock on the same day as the next time step for the same stock. On the other hand, grouping the data by stock would lead the LSTM to incorrectly assume that the next stock's data is a continuation of the previous stock's sequence. Essentially, the LSTM treats all the data as one continuous sequence, despite it actually consisting of 464 distinct sequences aggregated into a single dataset. To complicate matters further, there's very little information available on how to address this issue.

After thoroughly researching and studying the LSTM architecture, I devised a solution. As mentioned earlier, LSTMs retain context from previous inputs within a designated lookback period by storing this context in a matrix called the "hidden state." The issue arises when training the model on multiple stocks: the hidden state doesn't automatically reset between stocks, which means context from one stock could incorrectly influence predictions for another, even though the stocks are independent.

Fortunately, LSTMs offer the option to be either stateful or stateless. In a stateful LSTM, the hidden state persists across batches, while in a stateless LSTM, the hidden state resets after each batch. By setting the LSTM to be stateless, we can reset the context for each stock, preventing any information leakage between them.

The next step involves ensuring that the model trains on each stock's sequence separately. While we've addressed the hidden state issue, we still need to define clear boundaries between different stocks during training. This is where batch processing comes in. Normally, batching is used to process data in smaller, manageable chunks during training, improving efficiency and stability. However, in our case, we'll use batching to isolate each stock's data. By setting the batch size to match the number of sequences for a single stock and disabling batch shuffling, we ensure that the model processes each stock's data sequentially, transitioning from one stock to the next without mixing their contexts.

#### Class Imbalance
Before feeding data into our model, we need to address the issue of class imbalance. In stock data, it's common to have an uneven distribution of labels. For instance, our dataset contains significantly more instances of losses than profits. This imbalance can bias the model towards predicting losses more often, as it would be statistically more likely to be correct. To mitigate this, we can assign weights to each class, giving more importance to less frequent classes. This technique helps balance the influence of each class during training, encouraging the model to learn to recognize patterns associated with less common outcomes, such as profits. By doing so, we aim to improve the model's ability to make accurate predictions across all classes, not just the majority class.

#### Model Structure and Hyperparameters
The model is a Sequential LSTM-based network designed for multi-class classification of stock price movements. To enhance the model's ability to capture complex temporal patterns, we employ two LSTM layers, each with 150 units. This stacked architecture adds depth, allowing the model to better understand sequential dependencies in the data.

To prevent overfitting, each LSTM layer is followed by BatchNormalization, which normalizes layer inputs and helps stabilize learning, and Dropout set at 20%, which randomly omits neurons during training to promote generalization.

The model's output layer consists of a Dense layer with 4 units and a softmax activation function, providing probability distributions across the four classes. We will choose the class with the highest probability distribution as our output prediction. The model is compiled using the Adam optimizer, known for its adaptive learning rates, and sparse_categorical_crossentropy as the loss function, ideal for multi-class classification.

Early stopping is implemented to halt training if the validation loss doesn't improve for 5 consecutive epochs, ensuring the model retains the best weights and avoids overfitting.
