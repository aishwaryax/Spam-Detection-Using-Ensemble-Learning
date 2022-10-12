import sys
import os
import numpy as np
import os
from keras.models import Sequential
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten
from keras.layers import Conv1D,MaxPooling2D, \
    MaxPooling1D, Embedding, Dropout,\
    GRU,TimeDistributed,Conv2D,\
    Activation,LSTM,Input
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras import optimizers
from random import choice, randint

def optimizors(random_optimizor):
    if random_optimizor:
        i = randint(1,3)
        if i==0:
            opt = tf.keras.optimizers.SGD()
        elif i==1:
            opt= tf.keras.optimizers.RMSprop()
        elif i==2:
            opt= tf.keras.optimizers.Adagrad()
        elif i==3:
            opt = tf.keras.optimizers.Adam()
        elif i==4:
            opt =tf.keras.optimizers.Nadam()
        print(opt)
    else:
        opt= tf.keras.optimizers.Adam()
    return opt



def Build_Model_DNN(shape, nClasses, sparse_categorical,
                         min_hidden_layer_dnn, max_hidden_layer_dnn, min_nodes_dnn,
                         max_nodes_dnn, random_optimizor, dropout):
    """
    buildModel_DNN_Tex(shape, nClasses,sparse_categorical)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    layer = list(range(min_hidden_layer_dnn,max_hidden_layer_dnn))
    node = list(range(min_nodes_dnn, max_nodes_dnn))


    Numberof_NOde =  choice(node)
    nLayers = choice(layer)

    Numberof_NOde_old = Numberof_NOde
    model.add(Dense(Numberof_NOde,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        Numberof_NOde = choice(node)
        model.add(Dense(Numberof_NOde,input_dim=Numberof_NOde_old,activation='relu'))
        model.add(Dropout(dropout))
        Numberof_NOde_old = Numberof_NOde
    model.add(Dense(nClasses, activation='softmax'))
    model_tem = model
    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    return model,model_tem


def Build_Model_RNN(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical,
                         min_hidden_layer_rnn, max_hidden_layer_rnn, min_nodes_rnn, max_nodes_rnn, random_optimizor, dropout):
    model = Sequential()
    values = list(range(min_nodes_rnn,max_nodes_rnn))
    values_layer = list(range(min_hidden_layer_rnn,max_hidden_layer_rnn))

    layer = choice(values_layer)
    print(layer)
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    gru_node = choice(values)
    print(gru_node)
    for i in range(0,layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))

    model_tmp = model


    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    return model,model_tmp


def Build_Model_CNN(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical,
                       min_hidden_layer_cnn, max_hidden_layer_cnn, min_nodes_cnn, max_nodes_cnn, random_optimizor,
                       dropout, simple_model=False):
    model = Sequential()
    if simple_model:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                if len(embedding_matrix[i]) !=len(embedding_vector):
                    print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                     "into shape",str(len(embedding_vector))," Please make sure your"
                                     " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
        values = list(range(min_nodes_cnn,max_nodes_cnn))
        Layer = list(range(min_hidden_layer_cnn,max_hidden_layer_cnn))
        Layer = choice(Layer)
        for i in range(0,Layer):
            Filter = choice(values)
            model.add(Conv1D(Filter, 5, activation='relu'))
            model.add(Dropout(dropout))
            model.add(MaxPooling1D(5))

        model.add(Flatten())
        Filter = choice(values)
        model.add(Dense(Filter, activation='relu'))
        model.add(Dropout(dropout))
        Filter = choice(values)
        model.add(Dense(Filter, activation='relu'))
        model.add(Dropout(dropout))

        model.add(Dense(nclasses, activation='softmax'))
        model_tmp = model
        #model = Model(sequence_input, preds)
        if sparse_categorical:
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])
    else:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                if len(embedding_matrix[i]) !=len(embedding_vector):
                    print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                     "into shape",str(len(embedding_vector))," Please make sure your"
                                     " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)

                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        # applying a more complex convolutional approach
        convs = []
        values_layer = list(range(min_hidden_layer_cnn,max_hidden_layer_cnn))
        filter_sizes = []
        layer = choice(values_layer)
        print("Filter  ",layer)
        for fl in range(0,layer):
            filter_sizes.append((fl+2))

        values_node = list(range(min_nodes_cnn,max_nodes_cnn))
        node = choice(values_node)
        print("Node  ", node)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            #l_pool = Dropout(0.25)(l_pool)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs)
        l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
        l_cov1 = Dropout(dropout)(l_cov1)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
        l_cov2 = Dropout(dropout)(l_cov2)
        l_pool2 = MaxPooling1D(30)(l_cov2)
        l_flat = Flatten()(l_pool2)
        l_dense = Dense(1024, activation='relu')(l_flat)
        l_dense = Dropout(dropout)(l_dense)
        l_dense = Dense(512, activation='relu')(l_dense)
        l_dense = Dropout(dropout)(l_dense)
        preds = Dense(nclasses, activation='softmax')(l_dense)
        model = Model(sequence_input, preds)
        model_tmp = model
        if sparse_categorical:
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])


    return model,model_tmp
