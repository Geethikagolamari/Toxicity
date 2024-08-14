# Toxicity
Toxic comment recognition

## Abstract

The internet can be a mean and nasty place...but it doesn't need to be! Our goal will be to make the internet a safer place by recognizing and flagging such behavior. We are going to spot and detect toxic comments using Artificial Neural Networks and Python.

## Introduction

The artificial neural network we will be using to achieve our goal is RNN. We have designed the neural network using Python libraries like TensorFlow, Keras, Pandas, NumPy, etc. The dataset was obtained from the jigsaw toic comment classification challenge on Kaggle. It contained around 154,000 datapoints with each comment matched with six binary values.

## Theory and Algorithm

A recurrent neural network (RNN) is one of the two broad types of artificial neural networks, characterized by the direction of the flow of information between its layers. In contrast to the uni-directional feedforward neural network, it is a bi-directional artificial neural network, meaning that it allows the output from some nodes to affect subsequent input to the same nodes. Their ability to use internal state (memory) to process arbitrary sequences of inputs makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition.

The first step is a preprocessing step called tokenization which converts inputs to tokens(i.e. each word is converted into a sequence of integers). This is done using the inbuilt function in the keras function called vectorization( it also does a few additional pre-processing steps).

The next step is to pass these outputs through an embedding layer before they can be passed into the actual neural network. Embedding understands the features and attributes of the word and represents them in a vector of float or int values. The vector is determined by the location of the token in the embedding space where tokens with similar semantics are placed close to each other. 

Now we pass these embeddings through a deep neural network. We have used LSTM(Long short-term memory) cells as they are particularly good when it comes to sequences and NLP(Natural language processing) applications.

The six attributes we are trying to analyze from the comments are: 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'

The reasons for using the ReLU activation function are:
- Computation saving: 
- Deals with non-linearity
- Adds sparsity
- Mitigates vanishing gradient problem
  
The reason for choosing the sigmoid activation function for the output layer is because the function can deal with non-linearity and normalizes the output between 0 and 1.

The first dense layer helps to catch the non-linear relationships in the data after the bi-directional LSTM layer. The second dense layer further expands the model’s capacity to learn complex representations. The final dense layer allows the model to refine its learned representations further.

The Adam optimizer is a popular choice for training neural networks. It adapts the learning rates of each parameter during training, which can be beneficial for faster convergence and better handling of various types of data and architectures.

The model was trained for 1 epoch, 3 epochs, and 10 epochs but we found that the 3 epoch model was the most precise in it’s predictions. These models were trained on GPU and saved as .h5 files. The .h5 file was then used to make predictions offline for any given input.


## Dataset 
Kaggle: Toxic Comment Classification Challenge
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

## Observations and Conclusion

The project not only highlights the effectiveness of machine learning in addressing real-world issues related to online toxicity but also emphasizes the importance of ongoing research in this domain. As online communication continues to evolve, it is imperative to develop models that can adapt to emerging patterns of toxic behavior. Additionally, the ethical implications of deploying such models should be considered, ensuring that they do not inadvertently suppress legitimate freedom of expression.

Inferences drawn from the project suggest that further refinement and expansion of the dataset could potentially enhance the model's performance. Additionally, continuous monitoring and updating of the model will be necessary to ensure its effectiveness in a dynamic online environment. The successful implementation of the toxic comment classifier opens avenues for the application of similar techniques in addressing other challenges related to online content moderation and community well-being.

