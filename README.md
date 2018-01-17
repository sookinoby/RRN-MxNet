# Recurrent Neural Network using Apache MXNet

In our previous notebooks, we used a deep learning technique called Convolution Neural Network (CNN) to classify text and images. Although Convolutional neural network are powerful, they are designed to learn spatial features. They cannot learn temporal features from a input sequence such as audio, text. These types of neural network are called Feed Forward. Recurrent neural network (RNN) are type of neural network that can learn temporal features and has wider range of application than feed forward neural network.

In this notebook we will develop a recurrent neural network that predicts the probability of a word or character given the previous word or character. Almost all of us have a predictive keyboard on our smartphone, which suggests upcoming words for super-fast typing. Reccurent neural network allow us to build the most advanced predictive system similar to [SwiftKey](https://blog.swiftkey.com/swiftkey-debuts-worlds-first-smartphone-keyboard-powered-by-neural-networks/).

We will first cover the limitations of Feed Forward Neural Networks. Next, we will implement a basic RNN using Feed Forward Neural Network that can provide a good insight into how RNN works. Then we design a powerful RNN with LSTM and GRU layers using MxNet gluon API and use it to generate text.

We will also talk about the following topics:

1. Know the limitations of a Feed Forward Neural Network
2. Understand the idea behind RNN and LSTM
3. Install MXNet with Gluon API
4. Prepare datasets to train the Neural Network
5. Implement a basic RNN using Feed Forward Neural Network 
6. Implement an RNN Model to auto-generate text using Gluon API 

You need to have a basic understanding of Recurrent Neural Network(RNN), Activation Functions, Gradient Descent, Back Propagation and NumPy to understand this tutorial.

### The Need For Hidden State (memory)

Although Feed Forward Neural Networks, including Convolution Neural Networks, have shown great accuracy in classifying sentences and text, they cannot store long-term dependencies in memory (hidden state). For example, whenever an average American thinks about KFC chicken, her brain immediately thinks of it as "hot" and "crispy".
 ![Alt text](images/KFC_Thinking01.jpg?raw=true "Sequence to Sequence model")
 This is because our brains can remember the context of a conversation from memory, and retrieve those contexts whenever it needs. A Feed-Forward Neural Network can’t interpret the context. In a CNN can learn temporal context, a local group of neighbors within the size of its convolution kernels. So it cannot model sequential data (data with definitive ordering, like the structure of a language). An abstract view of feed-forward neural network is shown below <br /> ![Alt text](images/ffn_rnn.png?raw=true "Sequence to Sequence model")


An RNN is more versatile, it's cells accept weighted input and produce both weighted output (WO) and weighted hidden state (WH). The hidden state acts as the memory that stores context. If an RNN represents a person talking on the phone, the weighted output is the words spoken, and the weighted hidden state is the context in which the person utters the word.  ![Alt text](images/sequene_to_sequence.png?raw=true "Sequence to Sequence model") <br />

The yellow arrows are the hidden state, and the red arrows are the output.

A simple example can help us understand long term dependencies.

```python
<html>
<head>
<title>
RNN, Here I come.
 </title>
 </head> <body>HTML is amazing, but I should not forget the end tag.</body>
 </html>
 ```
Let’s say we are building a predictive text editor, which helps users auto-complete the current word by using the words in the current document and perhaps the users' prior typing habits.  The model should remember long-term dependencies like the need for the start tag <html> and end tag </html>. A CNN does not have provision to remember long term context like these. On the other hand, an RNN can remember the context using its internal "memory," just as a person might think “Hey, I saw an <html> tag, then a <title> tag, so I need to close the <title> tag before closing the <html> tag.”

### The intuition behind RNNs

Suppose we have to predict the 4th character in a stream of text, given the first three characters. To do that, we can design a simple Feed Forward Neural Network as in the following figure. ![Alt text](images/unRolled_rnn.png?raw=true "Unrolled RNN") <br />

This is basically a Feed Forward Network where the weights WI (green arrows) and WH (yellow arrows) are shared between some of the layers. This is an unrolled version of [Vanilla RNN](https://towardsdatascience.com/lecture-evolution-from-vanilla-rnn-to-gru-lstms-58688f1da83a), generally referred to as a many-to-one RNN because multiple inputs (3 characters, in this case) are used to predict one character. The RNN can be designed using MxNet as follows:

```python
class UnRolledRNN_Model(Block):
  # This is the initialisation of UnRolled RNN
    def __init__(self,vocab_size, num_embed, num_hidden,**kwargs):
        super(UnRolledRNN_Model, self).__init__(**kwargs)
        self.num_embed = num_embed
        self.vocab_size = vocab_size

        # Use name_scope to give child Blocks appropriate names.
        # It also allows sharing parameters between blocks recursively.
        with self.name_scope():
            self.encoder = nn.Embedding(self.vocab_size, self.num_embed)
            self.dense1 = nn.Dense(num_hidden,activation='relu',flatten=True)
            self.dense2 = nn.Dense(num_hidden,activation='relu',flatten=True)
            self.dense3 = nn.Dense(vocab_size,flatten=True)

    # This is the forward pass of neural network
    def forward(self, inputs):
        emd = self.encoder(inputs)
        #print(emd.shape)
        #since the input is shape(batch_size,input(3 characters))
        # we need to extract 0th,1st,2nd character from each batch
        chararcter1 = emd[:,0,:]
        chararcter2 = emd[:,1,:]
        chararcter3 = emd[:,2,:]
        c1_hidden = self.dense1(chararcter1) # green arrow in diagram for character 1 (WI)
        c2_hidden = self.dense1(chararcter2) # green arrow in diagram for character 2 (WI)
        c3_hidden = self.dense1(chararcter3) # green arrow in diagram for character 3 (WI)
        c1_hidden_2 = self.dense2(c1_hidden)  # yellow arrow in diagram (WH)
        addition_result = F.add(c2_hidden,c1_hidden_2) # Total c1 + c2
        addition_hidden = self.dense2(addition_result) # yellow arrow in diagram (WH)
        addition_result_2 = F.add(addition_hidden,c3_hidden) # Total c1 + c2 + c3
        final_output = self.dense3(addition_result_2)   # The red arrow in diagram (WO)
        return final_output
  ```
Basically, this neural network has 3 embedding layers (emb) for each character, followed by 3 dense layers:
Dense1 (with weights WI), which the input
Dense 2 (with weights WH) (an intermediate layer) 
Dense3 (with weights WO), which produces the output. We also do some MXNet array addition to combine inputs. 

In addition to the many-to-one RNN, there are other types of RNN that process such memory-based applications, including the popular sequence-to-sequence RNN:
![Alt text](images/loss.png?raw=true"Sequence to Sequence model") <br />


Here N inputs (3 characters) are mapped onto N outputs. This helps the model to train faster because we measure loss (the difference between the predicted value and the actual output) at each time instant. Instead of one loss at the end, we can see loss1, loss2, etc; So that we get a better feedback (backpropagation) when training our model.

We use [Binary Cross Entropy Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss) in our model.

This model can be folded back and succinctly represented like this:  <br />
 ![Alt text](images/RNN.png?raw=true "RNN")  <br />

The above representation also makes the math behind the model easy to understand:

```python
hidden_state_at_t = (WI x input + WH x previous_hidden_state)
```

There are some limitations with Vanilla RNN. For example, let’s say we have a long document has the sentences "I was born in France during the world war ….." and "So I can speak French." A Vanilla RNN cannot understand the context of being "born in France" and "I can speak French" if they can be far apart (temporally distant) in a given document.

RNN doesn’t provide the capability (at least in practice) to forget the irrelevant context in between the phrases. RNN gives more importance to the most previous hidden state because it cannot give preference to the arbitrary (t-k) hidden state, where t is the current time step and k is the number greater than 0. This is because training an RNN on a long sequence of words can cause the gradient to vanish (when the gradient is small) or to explode (when the gradient is large) during backpropagation. Basically, [backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html) multiplies the gradients along the computational graph in reverse direction. A detailed explanation of the problems with RNN is explained[here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.8930&rep=rep1&type=pdf).

### Long Short-Term Memory (LSTM)

To address the problems with Vanilla RNN, the two German researchers Sepp Hochreiter and Juergen Schmidhuber proposed [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (LSTM, a complex RNN unit) as a solution to the vanishing/exploding gradient problem.  A beautifully illustrated simpler version of LSTM can be found [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [here](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714). In an abstract sense, we can think LSTM unit as a small neural network that decides the amount of information it needs to preserve (memory) from the previous time step.

## Implementing an LSTM

Now we can try creating our own simple character predictor.

### Preparing your environment

If you're working in the AWS Cloud, you can save yourself a lot of installation work by using an [Amazon Machine Image](https://aws.amazon.com/marketplace/pp/B01M0AXXQB#support), pre-configured for deep learning.  If you have done this, skip steps 1-5 below.

If you are using a Conda environment, remember to install pip inside conda by typing 'conda install pip' after you activate an environment.  This will save you a lot of problems down the road.

Here's how to get set up:

1. Install [Anaconda](https://www.continuum.io/downloads), a package manager. It is easier to install Python libraries using Anaconda.
2. Install [scikit-learn](http://scikit-learn.org/stable/install.html), a general-purpose scientific computing library. We'll use this to pre-process our data. You can install it with 'conda install scikit-learn'.
3. Grab the Jupyter Notebook, with 'conda install jupyter notebook'.
4. Get [MXNet](https://github.com/apache/incubator-mxnet/releases), an open source deep learning library. The Python notebook was tested on version 0.12.0 of MxNet, and  you can install using pip as follows: pip install mxnet==0.12.0
5. After you activate the anaconda environment, type these commands in it: ‘source activate mxnet’

The consolidated list of commands are given below
```bash
conda install pip
pip install opencv-python
conda install scikit-learn
conda install jupyter notebook
pip install mxnet==0.12.0
```

6. You can download the MXNet notebook for this part of the tutorial [here](https://github.com/sookinoby/generative-models/blob/master/Test-rnn.ipynb), where we've created and run all this code, and play with it! Adjust the hyperparameters and experiment with different approaches to neural network architecture.

### Preparing the Data Set

We will use a work of [Friedrich Nietzsche](https://en.wikipedia.org/wiki/Friedrich_Nietzsche) as our dataset.
You can download the data set [here](https://s3.amazonaws.com/text-datasets/nietzsche.txt). You are free to use any other dataset, such as your own chat history, or you can download some datasets from this [site](https://cs.stanford.edu/people/karpathy/char-rnn/).

The dataset nietzsche.txt consists of 600901 characters, out of which 86 are unique. We need to convert the entire text to a sequence of numbers.

```python
chars = sorted(list(set(text)))
#maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
#maps indices to characters (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))
#convert the entire text into sequence
idx = [char_indices[c] for c in text]
```

### Preparing dataset for Unrolled RNN

Our goal is to convert the data set to a series of inputs and outputs. Each sequence of three characters from the input stream will be stored as the three input characters to our model, with the next character being the output we are trying to train our model to predict. For instance, we would translate the string "I_love_mxnet" into the following set of inputs and outputs. ![Alt text](images/unroll_input.png?raw=true "unrolled input") <br />

The code to do the conversion follows.

 ```python
 #Input for neural network(our basic rnn has 3 inputs, n samples)
cs=3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
#The output of rnn network (single vector)
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]
#Stacking the inputs to form3 input features
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

# Concatenate to form the input training set
col_concat = np.array([x1,x2,x3])
t_col_concat = col_concat.T

```
We also batchify the training set in batches of 32, so each training instance is of shape 32 X 3. Batchifying the input helps us train the model faster.

```python
#Set the batch size as 32, so input is of form 32 X 3
#output is 32 X 1
batch_size = 32
def get_batch(source,label_data, i,batch_size=32):
    bb_size = min(batch_size, source.shape[0] - 1 - i)
    data = source[i : i + bb_size]
    target = label_data[i: i + bb_size]
    #print(target.shape)
    return data, target.reshape((-1,))
```

### Preparing the dataset for gluon RNN

This is very similar to preparing the dataset for unrolled RNN, except for the shape of the input. The dataset should be ordered in the shape (number of example X batch_size). For example, let us consider the sample dataset below and batch it:

![Alt text](images/batch3.png?raw=true "batch reshape") <br />

In the above image, the input sequence is converted to a batch size of 3. By transforming it this way, we lose the temporal relationship between 'O' and 'V', 'M' and 'X'; but we can train our model faster in batches.

Below is the example with batch size 2 ![Alt text](images/batch4.png?raw=true "batch reshape") <br />.  It is very easy to generate the arbitrary length input sequence from a given batch. For example, if we want a sequence

 During our training, we use an input sequence length of 15. This is a hyperparameter and may require fine tuning for the best output.

### Designing RNN in Gluon

Next, we define a class that allows us to create two RNN models that we have chosen for our example: GRU (Gated Recurrent Unit)](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.rnn.GRU) and [LSTM](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.rnn.LSTM). GRU is a simpler version of LSTM and performs equally well. You can find a comparison study [here](https://arxiv.org/abs/1412.3555). The models are created with the following Python snippet:


```python
# Class to create model objects.
class GluonRNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, **kwargs):
        super(GluonRNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))

            if mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden
   
 #define the forward pass of the neural network
    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden
    #Initial state of network
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```
The constructor of the class creates the neural units that will be used in our forward pass. The constructor is parameterized by the type of RNN layer (LSTM, GRU or Vanilla RNN) to use.  The forward pass method will be called when training the model to generate the loss associated with the training data.

The forward pass function starts by creating an [embedding layer](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.nn.Embedding) for the input character. You can look at our [previous blog post](https://www.oreilly.com/ideas/sentiment-analysis-with-apache-mxnet) for more details on embedding. The output of the embedding layer is provided as input to the RNN. The RNN returns an output as well as the hidden state. There is dropout layer to prevent overfitting so that the model doesn’t memorize the input-output mapping.   The output produced by the RNN is passed to a decoder (dense unit), which predicts the next character in the neural network and also generates the loss during the training phase.

We also have a “begin state” function that initialises the initial hidden state of the model.

### Training the neural network

After defining the network, now, we have to train the neural network so that it learns.

```python
def trainGluonRNN(epochs,train_data,seq=seq_length):
    for epoch in range(epochs):
        total_L = 0.0
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, seq_length)):
            data, target = get_batch(train_data, i,seq)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target) # this is total loss associated with seq_length
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and seq_length to balance it.
            gluon.utils.clip_global_norm(grads, clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()
```

Each epoch starts by initializing the hidden units to zero. While training each batch, we detach the hidden unit from computational graph so that we don’t backpropagate the gradient beyond the sequence length (15 in our case). If we don’t detach the hidden state, the gradient is passed to the beginning of hidden state (t=0). After detaching, we calculate the loss and use the backward function to back-propagate the loss in order to fine tune the weights. We also normalize the gradient by multiplying it by the sequence length and batch size.

### Text generation

After training for 200 epochs, we can generate random text. The weights of the trained model are available [here](https://www.dropbox.com/s/7b1fw94s1em5po0/gluonlstm_2?dl=0). You can download the model parameters and load it using [model.load_params](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=load#mxnet.module.BaseModule.load_params) function.

To generate text, we initialize the hidden state.
```python
 hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=context)
```
Remember, we don't have to reset the hidden state as we don’t backpropagate the loss (fine tune the weights).


Then, we reshape the input sequence vector to a shape that the RNN model accepts.

```python
 sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T
                                ,ctx=context)
```

Then we look at the argmax of the output produced by the network. generate output char 'c'.

```python
output,hidden = model(sample_input,hidden)
output,hidden = model(sample_input,hidden)
index = mx.nd.argmax(output, axis=1)
index = index.asnumpy()
count = count + 1
```

Then append output char 'c' to input string

```python
sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T,ctx=context)
new_string = new_string + indices_char[index[-1]]
input_string = input_string[1:] + indices_char[index[-1]]
```

Next, slice the first character of the input string.

```python
 new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
```


```python
# a nietzsche like text generator
import sys
def generate_random_text(model,input_string,seq_length,batch_size,sentence_length_to_generate):
    count = 0
    new_string = ''
    cp_input_string = input_string
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=context)
    while count < sentence_length_to_generate:
        idx = [char_indices[c] for c in input_string]
        if(len(input_string) != seq_length):
            print(len(input_string))
            raise ValueError('there was a error in the input ')
        sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T
                                ,ctx=context)
        output,hidden = model(sample_input,hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1
        new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
    print(cp_input_string + new_string)

```

If you look at the text generated, we will note the model has learnt open and close quotations(""). It has a definite structure and looks similar to 'nietzsche'.

Next, we will take a look at generative models*, specially Generative Adversial Network  a powerful model that can generate new data from a given input dataset.

*Note - Although RNN model is used to generate text, it is not actually a 'Generative Model' in the strict sense. This [pdf document]((https://arxiv.org/pdf/1703.01898.pdf) clearly illustrates the difference between a  generative model and discriminative model for text classification.

