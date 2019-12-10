import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from rbf_keras.rbflayer import RBFLayer, InitCentersRandom

def load_data():
    '''
    Returns training and testing data as dictionaries of numpy arrays.
    tf.data datasets are not supported by the current rbf_keras implementation.
    '''

    ( x_train, y_train), ( x_test, y_test ) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape( -1, 784 ), x_test.reshape( -1, 784 )

    y_train, y_test = tf.one_hot( y_train, 10 ).numpy(), tf.one_hot( y_test, 10 ).numpy()


    return {'x': x_train, 'y': y_train}, {'x': x_test, 'y': y_test}

def create_model(rbf_neurons=100):
    model = Sequential()
    model.add( Flatten() )
    rbflayer = RBFLayer(rbf_neurons, betas=0.1, input_shape=(784,1),
            initializer=InitCentersRandom(train_ds['x']) ) 
    model.add( rbflayer )
    model.add( Dense( 10, activation='tanh',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-04, l2=1e-04 )))
    model.add( Dense( 10, activation='softmax',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-04, l2=1e-04 )))

    return model


train_ds, test_ds = load_data()
model = create_model()

optm_obj=Adam()
loss_obj = CategoricalCrossentropy()
acc_metric = tf.keras.metrics.CategoricalAccuracy( name='accuracy_metric' )
model.compile( loss=loss_obj, optimizer=optm_obj, metrics=[acc_metric] )

model.fit( train_ds['x'], train_ds['y'],
        batch_size=64, epochs=2000, validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)] )

acc_metric.reset_states()
y_pred = model.predict( train_ds['x'] )
acc_metric.update_state( train_ds['y'], y_pred )
print( 'joint training and validation accuracy: {}'.format( acc_metric.result() ) )

acc_metric.reset_states()
y_pred = model.predict(test_ds['x'])
acc_metric.update_state( test_ds['y'], y_pred )
print( 'testing accuracy: {}'.format( acc_metric.result() ) )
