# RNNs


## Project 1: [LSTM-next-app](https://github.com/pedrofuentes79/RNNs/tree/master/LSTM-next-app)

This model predicts 8 apps that the user will use next based on the user's previous app usage history.
The dataset is from kaggle. It can be found [here](https://www.kaggle.com/datasets/johnwill225/daily-phone-usage)

## Project 2: [Sentiment-Analysis](https://github.com/pedrofuentes79/RNNs/tree/master/Sentiment-Analysis)  
This model predicts the sentiment of a product review. The article which explains the model can be found [here]().
The dataset is from kaggle. It can be found [here](https://www.kaggle.com/datasets/jillanisofttech/amazon-product-reviews). It consists of amazon product reviews, which have text and a score from 1 to 5. These scores are mapped to 0 (negative) and 1 (positive). The model is trained on the text and the score is used as the label. The model is then used to predict the sentiment of a review.
In the (Sentiment-Analysis directory)[https://github.com/pedrofuentes79/RNNs/tree/master/Sentiment-Analysis] there are several implementations of different models to solve this problem, including a CNN, a RNN with LSTM layers and a BERT model fine tuned with a small sample of the dataset.   
There are some variants to the implementations, for example there is a CNN model which uses a pretrained word2vec model to initialize the embedding layer and there is another one which uses keras' embedding layer.

## Packages
* [Tensorflow](https://www.tensorflow.org/)
* [Sci-kit Learn](https://scikit-learn.org/stable/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Gensim](https://radimrehurek.com/gensim/)
* [NLTK](https://www.nltk.org/)



