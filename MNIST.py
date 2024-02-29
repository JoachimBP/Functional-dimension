from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
from time import time
from sklearn.model_selection import train_test_split

from keras.datasets import mnist

from Functions.Functions import functional_dimension



import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Download MNIST data
(X_1, y_1), (X_2, y_2) = mnist.load_data()
X_1 = X_1.reshape(60000, 784)
X_2 = X_2.reshape(10000, 784)
X_1 = X_1.astype('float32')
X_2 = X_2.astype('float32')
X_1 /= 255
X_2 /= 255

# Mix train and test data
X = np.concatenate((X_1, X_2), axis =0)
y = np.concatenate((y_1, y_2))

# Choose the desired sizes for train and test
train_size = 600
test_size = 2000

# Randomly split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = train_size)
X_test = X_test[0:test_size,:]
y_test = y_test[0:test_size]

# Randomly generate gaussian inputs
random_size = test_size
X_random = np.random.normal(random_size*784*[0.5],1).reshape(random_size, 784)




    ##



w_list = []

train_spec = []
test_spec = []
random_spec = []

accuracy = []
val_accuracy = []
loss = []
val_loss = []
computation_time = []
epoch = []



t0 = time()

# Creation of the network
w = 30   # Width of the network

N0 = 784
N1 = w
N2 = w
N3 = w
NS = 10

Arch = [N0, N1, N2, N3, NS]
L = len(Arch) - 1

model = Sequential()
model.add(Dense(N1,  input_dim=N0))
model.add(Activation('ReLU'))
model.add(Dense(N2))
model.add(Activation('ReLU'))
model.add(Dense(N3))
model.add(Activation('ReLU'))
model.add(Dense(NS))
model.add(Activation('softmax'))

# Number of parameters and number of identifiable parameters (i.e. maximum theoretical functional dimension)
Npar = model.count_params()
Npar_identif = Npar - np.sum(Arch) + Arch[0] + Arch[L]

# Parameters of the training
from keras.optimizers import SGD
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# Convert class vectors to binary class matrices
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

batch_size = 256 

# Epoch increment, i.e. number of SGD epochs between two computations
# This number increases during training
nb_epoch_1 = 40
nb_epoch_2 = 200
nb_epoch_3 = 400
nb_iter_1 = 2
nb_iter_2 = 1
nb_iter_3 = 1
nb_iter = nb_iter_1 + nb_iter_2 +  nb_iter_3
total_epoch = nb_epoch_1 * nb_iter_1 + nb_epoch_2 * nb_iter_2 + nb_epoch_3 * nb_iter_3


nb_epoch = nb_epoch_1   # Current epoch increment

for cpt in range(nb_iter):
    eprint('Iteration', cpt)

    # After a certain number of iterations, change the epoch increment
    if cpt == nb_iter_1:   
        nb_epoch = nb_epoch_2
    if cpt == nb_iter_1 + nb_iter_2:
        nb_epoch = nb_epoch_3
    t0 = time()

    # Train the network
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,verbose=0,validation_data = (X_test, Y_test))

    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    eprint('acc:', acc, '- val_acc:', val_acc, '- loss:', loss, '- val_loss:', val_loss)

    # Store the current total number of epochs
    if cpt <  nb_iter_1:
        epoch.append((cpt + 1) * nb_epoch_1)
    elif cpt < nb_iter_1 + nb_iter_2:
        epoch.append(nb_iter_1 * nb_epoch_1 + (cpt + 1 - nb_iter_1) * nb_epoch_2)
    else:
        epoch.append(nb_iter_1 * nb_epoch_1 + nb_iter_2 * nb_epoch_2 + (cpt + 1 - nb_iter_1 - nb_iter_2) * nb_epoch_3)

    # Store the accuracies and losses
    accuracy.append(acc)
    val_accuracy.append(val_acc)
    loss.append(loss)
    val_loss.append(val_loss)



  
    # Compute the batch functional dimension for the trained model and several input choices
    fdim = functional_dimension(model)

    ## X_train
    fdim.get_differential(X_train)
    spec = fdim.compute_svd()
    train_spec.append(spec)

    ## X_test
    fdim.get_differential(X_test)
    spec = fdim.compute_svd()
    test_spec.append(spec)

    ## X_random
    fdim.get_differential(X_random)
    spec = fdim.compute_svd()
    random_spec.append(spec)

    
    ##

    t1 = time()

    comp_time = round((t1 - t0)/60, 2)

    eprint('Computation time:', comp_time)
    computation_time.append(comp_time)


    ##


# Store the lists of computed spectra in separate files for further analysis
import pickle
pickle.dump(train_spec, open('train_spec.dat', 'wb'))
pickle.dump(test_spec, open('test_spec.dat', 'wb'))
pickle.dump(random_spec, open('random_spec.dat', 'wb'))



print('Batch size:', batch_size, ' - Total number of epochs:', total_epoch)
print('Architecture =', Arch)
print('Number of parameters:', Npar)
print('Number of identifiable parameters:', Npar_identif)

print('Train size:', train_size, '- Test_size:', test_size, '- Random size:', random_size)
print('Epochs =', epoch)
print('Final_train_loss =', loss)
print('Final_test_loss =', val_loss)
print('Final_train_accuracy =', accuracy)
print('Final_test_accuracy =', val_accuracy)
print('Computation_time =', computation_time)