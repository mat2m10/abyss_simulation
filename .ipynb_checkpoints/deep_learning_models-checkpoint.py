from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from tensorflow.keras import Input, Model, layers, regularizers
from tensorflow.keras.layers import Input, Dense

def abyss(geno, bottleneck_nr, epoch, patience):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(geno, geno, test_size=0.2, random_state=42)
    
    # Regularization parameter
    l2_regularizer = 0.001
    
    # Original autoencoder model with L2 regularization
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Dense(bottleneck_nr, activation='elu', name='bottleneck', input_shape=(geno.shape[1],), kernel_regularizer=regularizers.l2(l2_regularizer)),  # Bottleneck layer with L2 regularization
        layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(geno.shape[1], activation='tanh', kernel_regularizer=regularizers.l2(l2_regularizer))
    ])
    
    # Compile the original model with L2 regularization
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mean_absolute_error'])
    
    # Define Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Fit the original model with Early Stopping
    history = autoencoder.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    # Extract the bottleneck layer after fitting the model
    bottleneck_model = tf.keras.Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer('bottleneck').output)
    
    return autoencoder, bottleneck_model, history
    
# deep abyss
def deep_abyss(geno, bottle, epoch, patience, pheno):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, pheno_train, pheno_test = train_test_split(geno, geno, pheno, test_size=0.2, random_state=42)

    # Define your regularization strength (lambda)
    l2_lambda = 0.001  # Adjust this value as needed

    # Define input layers
    input_shape_geno = geno.shape[1:]
    input_layer_geno = Input(shape=input_shape_geno, name='input_geno')

    input_shape_pheno = pheno.shape[1:]
    input_layer_pheno = Input(shape=input_shape_pheno, name='input_pheno')

    # Create layers
    encoder_init_1 = Dense(bottle, 
                           activation="elu", 
                           name="encoder_init_1",
                           kernel_regularizer=regularizers.l2(l2_lambda))
    
    decoder_init_2 = Dense(input_shape_geno[0], 
                           activation="tanh", 
                           name="decoder_init_2",
                           kernel_regularizer=regularizers.l2(l2_lambda))
    
    predictor = Dense(input_shape_pheno[0], 
                           activation="linear", 
                           name="predictor",
                           kernel_regularizer=regularizers.l2(l2_lambda))

    # Define custom layer for element-wise trainable weights
    class ElementWiseWeightsLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(ElementWiseWeightsLayer, self).__init__(**kwargs)
    
        def build(self, input_shape):
            self.weight = self.add_weight(shape=(), initializer="ones", trainable=True, name="element_wise_weight")
            super(ElementWiseWeightsLayer, self).build(input_shape)
    
        def call(self, inputs):
            return inputs * self.weight
    
    # Define encoder and decoder paths
    bottle_neck = encoder_init_1(input_layer_geno)
    allele_frequency_probability = decoder_init_2(bottle_neck)
    y_predictor = predictor(allele_frequency_probability)
    
    # Define the model
    autoencoder = Model(inputs=input_layer_geno, outputs=[allele_frequency_probability, y_predictor], name="fishy")

    # Compile the model
    autoencoder.compile(optimizer='adam', loss=['mse', 'mse'], loss_weights=[2.0, 1.0])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Train the model
    history = autoencoder.fit(X_train, [X_train, pheno_train], epochs=epoch, batch_size=32, validation_data=(X_test, [X_test, pheno_test]), callbacks=[early_stopping], verbose=0)

    # Extract the bottleneck layer
    bottleneck_model = tf.keras.Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer('encoder_init_1').output)
    return autoencoder, bottleneck_model, history