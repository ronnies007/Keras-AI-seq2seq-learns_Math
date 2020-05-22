# code taken from here: https://github.com/lukas/ml-class/blob/master/videos/seq2seq/train.py
# original source seems to be: https://keras.io/examples/nlp/addition_rnn/

from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint, RemoteMonitor, LambdaCallback, TensorBoard
import numpy as np
import NN_config as cfg
import random
import math
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
from visual_callbacks import AccLossPlotter

# The GPU id to use, usually either "0" or "1";
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output """
    def __init__(self, chars):
        """Initialize character table. # Arguments
            chars: Characters that can appear in the input. """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C. # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                      used to keep the # of rows for each data the same. """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)




# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of int is DIGITS.  <-- not anymore since multiplikation 
# needs even more digits!
add = 10
for n in range(1,cfg.digits*2):  # variable 'add' for reserving enough digits for multiplikation
    add = add * 10
addMe = len(str(add)) - cfg.digits
print("addMe:", addMe)
maxlen = cfg.digits + addMe + cfg.digits

# All the numbers, plus sign and space for padding.
chars = '0123456789+-*/. '
ctable = CharacterTable(chars)

# --- creates a questions/answers set in form of strings e.g. q='123-5' a'=118' ----
def send_new_data(questions, expected):
    print('Generating data...')
    questions = []
    expected = []
    seen = set()
    while len(questions) < cfg.training_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789'))
                        for i in range(np.random.randint(1, cfg.digits + 1))))              
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        answer = ''
        query = ''
        q = ''
        # Pad the data with spaces such that it is always MAXLEN.
        while (len(q) < cfg.digits):
            choice = random.randint(1,4)
            if (choice == 1):           # addition
                q = '{}+{}'.format(a, b)
                answer = str(a + b)
                query = q + ' ' * (maxlen - len(q))             # padding the question 
            if (choice == 2):           # multiplikation
                q = '{}*{}'.format(a, b)
                answer = str(a * b)
                query = q + ' ' * (maxlen - len(q))             # padding the question 
            if (choice == 3):           # subtraktion
                q = '{}-{}'.format(a, b)
                answer = str(a - b)
                query = q + ' ' * (maxlen - len(q))             # padding the question 
            if (choice == 4):           # division
                while (b == 0) or (a == 0):                     # to hopefully prevent division by zero
                    a, b = f(), f()
                q = '{}/{}'.format(a, b)
                answer = str(round(a / b , 2))
                query = q + ' ' * (maxlen - len(q))             # padding the question 
            # answers can be of maximum size DIGITS + addme.
            answer += ' ' * (cfg.digits + addMe - len(answer))  # padding the answer
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. turned off 
        if cfg.reverse:
            query = query[ : : -1 ]
        
        questions.append(query)
        expected.append(answer)
        print("training_samples generated:", len(questions), end="\r")

    print('\n')

    # print('Vectorization...')
    x = np.zeros((len(questions), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), cfg.digits + addMe, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, maxlen)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, cfg.digits + addMe)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 20
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    return x_train, y_train, x_val, y_val, questions, expected

# look for already saved models in specified folder
save_model_exist = False
if (os.path.exists(cfg.checkpoint_path)):            # if the models are named correctly it will always pick the last saved. 
    for r, d, files in os.walk(cfg.checkpoint_path): # it picks '9' instead of '19' so better name it '000009' and '000019' 
        for f in files:                          # then it should work.
            ext = f[len(f)-4:]
            if(ext == 'hdf5'):
                print("found saved model:", cfg.checkpoint_path + f)
                save_model_exist = True

if (save_model_exist):
    print("loading model from:", cfg.checkpoint_path + f)
    # Recreate the exact same model, including its weights and the optimizer
    model = load_model( cfg.checkpoint_path + f )
    # Show the model architecture
    model.summary()
else:
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length, use input_shape=(None, num_feature).
    model.add(LSTM(cfg.hidden_size_1, input_shape=(maxlen, len(chars)))) 
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'DIGITS + addMe' times as that's the maximum length of output, 
    # (e.g., when DIGITS=3, max output is 999+999=1998.) --> this is not true anymore since I added multiplikation, 
    # which needs more space
    model.add(RepeatVector(cfg.digits + addMe))
    model.add(LSTM(cfg.hidden_size_2, return_sequences=True)) 
    # model.add(Dropout(0.1))
    model.add(LSTM(cfg.hidden_size_2, return_sequences=True))
    model.add(Dropout(0.1))
    # Apply a dense layer to the every temporal slice of an input. 
    # For each of step of the output sequence, decide which character should be chosen.
    model.add(TimeDistributed(Dense(len(chars), activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()


if os.path.exists(cfg.log_path):
    if not os.path.exists(cfg.log_path + cfg.log_file):     
        # WRITE A BRANDNEW Logfile
        print(cfg.log_path + cfg.log_file, "not found.")
        print("new logfile is being created... @", cfg.log_path)
        try:
            cfg.logfile3 = open(cfg.log_path + cfg.log_file,"w")
            cfg.logfile3.write("time,batch,loss,accuracy")
            cfg.logfile3.write("\r\n") 
            cfg.logfile3.close() 
            time.sleep(2)
        except: 
            print ("logfile path does not exist.")
    else:
        print("--> logs will be appended to existing logfile")
        df = pd.read_csv(cfg.log_path + cfg.log_file)
        cfg.step_counter = df['batch'].max()
        makeNew = False


def on_batch_end(batch, logs):
    logTime = str(time.strftime("%d.%m.%Y_%H:%M:%S"))
    cfg.logfile3=open(cfg.log_path + cfg.log_file,"a") #write loginfo to windows txt 
    cfg.logfile3.write(str(logTime))
    cfg.logfile3.write(",")
    cfg.logfile3.write(str(cfg.step_counter))
    cfg.logfile3.write(",")
    cfg.logfile3.write(str(logs['loss']))
    cfg.logfile3.write(",")
    cfg.logfile3.write(str(logs['accuracy'])) 
    cfg.logfile3.write("\n") 
    cfg.logfile3.close() 
    cfg.step_counter += 1


def on_epoch_end(epoch, logs):
    
    # Function invoked at end of each epoch. only to hold track of the actual epochs,
    # because training runs in custom loop with epochs=1
    cfg.epoch_count += 1 # {epoch:02d}_{val_loss:.2f}
    print("\n")
    print('--------- Epoch: ' + str(cfg.epoch_count) + '/' + str(cfg.max_epochs) + ' ----------')
    print(' questions   truth    predicted  ')


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
batch_callback = LambdaCallback(on_batch_end=on_batch_end) # save_weights_only=True

checkpoint_save = ModelCheckpoint(str(cfg.checkpoint_path) + "checkpoint_model_00000" + str(cfg.epoch_count) + ".hdf5", 
                                            save_best_only=True, monitor='val_loss', mode='auto', period=3)
                                                                # period=3 --> how many epochs in between saves 
callbacks = []
callbacks.append(checkpoint_save)
callbacks.append(print_callback)
callbacks.append(batch_callback)

# init data first time
x_train, y_train, x_val, y_val, questions, expected = send_new_data([], []) # generate new questions

# train    
for iteration in range( 1, cfg.max_epochs ):
    model.fit(x_train, y_train,
        batch_size=cfg.batch_size,
        epochs=1,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks)


    # Select 5 samples from the validation set at random so we can visualize errors.
    for i in range(5):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if cfg.reverse else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('☑', end=' ')
        else:
            print('☒', end=' ')
        print(guess)

    
    x_train, y_train, x_val, y_val, questions, expected = send_new_data([], []) # generate new questions and answers after every 1 epoch

# fini
time_now = datetime.now().strftime("%m_%d_%Y_%H_%M")
model.save(cfg.checkpoint_path + "checkpoint_model_" + time_now + "_final.hdf5")
exit()
