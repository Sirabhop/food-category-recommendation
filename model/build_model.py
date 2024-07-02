from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

def build_model(n_dish_type):
    model_url = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    embed = hub.KerasLayer(model_url, trainable=True, name='USE_embedding')
    
    model = Sequential([
        # Input layer
        Input(shape=[], dtype=tf.string),

        # Embedding layer
        embed,

        # MLP layer
        Dropout(0.1),

        # Output layer
        Dense(n_dish_type, activation='softmax')
    ])
    
    model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
    er_stop = EarlyStopping(monitor='val_loss', patience=5)
    
    callbacks = [reduce_lr, er_stop]
    
    return model, callbacks