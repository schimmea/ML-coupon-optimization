import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, LeakyReLU, concatenate, Input, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
import h5py

# This file is for building the main model, replicated closely following Gabel & Timoshenko (2020)

def build_model(J, T, H, L, learning_rate=0.001):
    # Inputs:
    #   J:              Number of products
    #   T:              Number of weeks
    #   H:              Number of convolutional filters
    #   L:              Size of bottleneck
    #   learning_rate:  Adam learning rate
    
    # Output:
    #   Model, ready for fitting
    
    # Purchase Histories until t
    B_T = Input(shape=(J, T), name="hists", dtype=tf.float32) 
    # Coupons received at t+1  
    D = Input(shape=(J, 1), name="coups", dtype=tf.float32)  
    # Make the coupons percentages to match the other input domains
    D_perc = tf.math.scalar_mul(0.01, D)
    
    # Infer purchase frequencies from B_T by calculating the mean over all weeks
    B_inf = tf.math.reduce_mean(B_T, axis=2, keepdims=True, name="freqs")
    
    # Apply convolutional filter and leaky ReLU to histories
    B_H = Conv1D(filters=H, kernel_size=1, use_bias=False)(B_T)  # kernel_size specifies the rows that the filter spans
                                                                 # it always spans the entire width
    B_H = LeakyReLU(alpha=0.2)(B_H)

    # Encode all the matrices:
    # We need to transpose the tensors first, because keras does matmul(input, weights)
    # Pytorch gives a little more freedom and allows matmul(weights, input)
    # The result is the same
    B_H_bar = Permute((2,1), name="Transpose_B_H")(B_H)
    # Afterwards, create an encoding layer...
    B_H_bar = Dense(units=L, activation="linear", use_bias=False, name="W_IN_B_H")(B_H_bar)
    # ... and a decoding layer
    B_H_bar = Dense(units=J, activation="linear", use_bias=False, name="W_OUT_B_H")(B_H_bar)
    B_H_bar = Permute((2,1), name="Transpose_B_H_bar_back")(B_H_bar)

    # Repeat for the other inputs
    B_inf_bar = Permute((2,1), name="Transpose_B_inf")(B_inf)
    B_inf_bar = Dense(units=L, activation="linear", use_bias=False, name="W_IN_B_inf")(B_inf_bar)
    B_inf_bar = Dense(units=J, activation="linear", use_bias=False, name="W_OUT_B_inf")(B_inf_bar)
    B_inf_bar = Permute((2,1), name="Transpose_B_inf_bar_back")(B_inf_bar)

    D_bar = Permute((2,1), name="Transpose_D")(D_perc)
    D_bar = Dense(units=L, activation="linear", use_bias=False, name="W_IN_D")(D_bar)
    D_bar = Dense(units=J, activation="linear", use_bias=False, name="W_OUT_D")(D_bar)
    D_bar = Permute((2,1), name="Transpose_D_bar_back")(D_bar)
    
    # Concatenate the inputs and outputs:
    first_col = tf.ones_like(D, name="Ones_column_of_concat")
    z = concatenate([first_col, B_H, B_H_bar, B_inf, B_inf_bar, D_perc, D_bar], axis=2)

    # Get probabilities
    probs = Dense(units=1, activation="sigmoid", name="Sigmoid", use_bias=False)(z)

    model = Model(inputs=[B_T, D], outputs=probs)
    
    # Get the pretrained P2V Embeddings (Gabel, Guhl, Klapper, 2019), see Product2Vec folder
    with h5py.File('Product2Vec/data/p2v_product_vectors.h5', 'r') as hf:
        p2v_weights = hf['p2v_product_embeddings'][:]
    
    # Initialize the W_IN matrices with the embeddings
    model.layers[9].set_weights([p2v_weights])
    model.layers[10].set_weights([p2v_weights])
    model.layers[11].set_weights([p2v_weights])
    
    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate), metrics=[AUC()])
    
    return model
    