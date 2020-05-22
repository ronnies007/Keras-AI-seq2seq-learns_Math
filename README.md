# Keras-AI-seq2seq-learns_Math
seq2seq model in keras extended from addition to all four basic computations (+, -, *, /)

# code taken from here: https://github.com/lukas/ml-class/blob/master/videos/seq2seq/train.py
# original source seems to be: https://keras.io/examples/nlp/addition_rnn/

- u need to define log- and checkpoint- path as well as log_file name at the top of the code.
- at the end of each epoch u will get validation examples. to see how good it got so far.
  also the model will be saved at this point. as it will be at the end of training.
- there is a NN_config.py which is just empty. Since I dont like using the keras FLAG stuff, this is my way to get global
  variables like path etc. using the global character of the imported variables.
- during training there is a custom logfile written. it needs to be opened manually from another console. 
  then its possible to watch the learning curves. (the matplotlib code is messy. sry, its old. I just quickly adapted
  it for this project. some functions like zooming in and staying there, are missing.)
  
- new training batches are being created before every epoch now, ..my conclusion is, that way, it cannot overfit so easily.
- the current model takes quite some time to train. I am still experimenting with 'val_accuracy=0.9200' atm.
  so not every answer is correct. in my eyes it needs val_accuracy 0.99 at least to be trusted.


whats missing is user input to manually type in questions and hit enter for answer(s).
feel free to test. have fun.

  
