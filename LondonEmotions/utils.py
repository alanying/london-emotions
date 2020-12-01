import time
from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.python.lib.io import file_io
import numpy as np

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with file_io.FileIO(filepath, mode='r') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

def instantiate_model(embedd_matrix, max_seq_len, vocab_size, embed_num_dims):

    embedd_layer = Embedding(vocab_size,
                             embed_num_dims,
                             input_length = max_seq_len,
                             weights = [embedd_matrix],
                             trainable=False)
    # Convolution
    kernel_size = 3
    filters = 256

    model = Sequential()
    model.add(embedd_layer)
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(5, activation='softmax'))


    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
