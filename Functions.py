import tensorflow as tf
from keras.models import Sequential
import time
import numpy as np
import torch

def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    # create an empty model
    new_model = Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model 

class functional_dimension():

    def __init__(self, model, use_logits = 0, verbose = 0): 

        self.model = model
        self.use_logits = use_logits

        layers = model.layers
        l = len(layers)
        if use_logits == 1:
            model = extract_layers(model,0,l-2)
            self.model = model

        
        # Find the architecture of the model
        self.layers = model.layers
        l = len(self.layers)
        L = int((l + 1)/2)
        self.number_layers = L
        N0 = model.get_weights()[0].shape[0]
        self.arch = [N0]
        for i in range(L):
            self.arch.append(self.layers[2*i].units)

        # Number of parameters of the model
        self.Npar = model.count_params()

        # Number of identifiable parameters of the model
        self.Npar_identif = self.Npar - np.sum(self.arch) + self.arch[0] + self.arch[L]


        # Function computing the gradients
        @tf.function
        def jacob(x):
            with tf.GradientTape() as tape:
                y = model(x)
            model_gradients = tape.jacobian(y, model.trainable_variables)
            return model_gradients
        
        self.jacob = jacob


        if verbose != 0:
            print("Number of layers: L =", L)
            print("Architecture:", self.arch)
            print("Number of parameters:", self.Npar)
            print('Number of identifiable parameters:', self.Npar_identif)
        
        
    def get_differential(self, X, verbose = 0):
        
        self.stored_svd = 0
        self.stored_norm = 0
        self.stored_frob = 0
        Nsample = X.shape[0]

        if verbose != 0:
            print('Size of the sample: ', Nsample)

        jacob = self.jacob

        t0 = time.time()

        arch = self.arch
        L = self.number_layers

        # Creation of the matrix that will be the jacobian
        Gamma = np.zeros((Nsample*arch[L],self.Npar))
    
        batch_size = min(1000,Nsample,self.Npar_identif)    # We are going to complete Gamma one batch of inputs at a time

        i = 0   # input index
        while i <= Nsample-batch_size:
            line = i * arch[L]    # line index for Gamma

            model_gradients = jacob(X[i:i+batch_size])

            column = 0     # column index for Gamma

            # Gradients with respect to the weights
            for cpt in range(L):
                Gamma[line:line+batch_size*arch[L],column:column+arch[cpt]*arch[cpt+1]] = model_gradients[2*cpt].numpy().reshape(batch_size*arch[L],arch[cpt]*arch[cpt+1])
                column += arch[cpt]*arch[cpt+1]
            
            # Gradients with respect to the biases
            for cpt in range(L):
                Gamma[line:line+batch_size*arch[L],column:(column+arch[cpt+1])] = model_gradients[2*cpt+1].numpy().reshape(batch_size*arch[L],arch[cpt+1])
                column += arch[cpt+1]
            i += batch_size
        
        t1 = time.time()

        if verbose != 0:
            print('Computation time:', (t1 - t0)/60)

        self.Gamma = Gamma

        return Gamma


    def compute_svd(self, use_torch = 1, verbose = 0):

        Gamma = self.Gamma
        t0 = time.time()
        if self.stored_svd == 0:
            if use_torch == 1:
                Gamma_torch = torch.from_numpy(Gamma)
                svd = torch.linalg.svdvals(Gamma_torch)
                svd = svd.numpy()
            else:
                svd = np.linalg.svd(Gamma, compute_uv=False)  
            t1 = time.time()
            self.svd = svd
            self.stored_svd = 1
        
        svd = self.svd

        if verbose != 0:
            print('SVD:', svd)
            print('Computation time:', (t1 - t0)/60)
        
        return svd
    
    
  
    def compute_rank(self, use_torch = 1, verbose = 0):
        
        Gamma = self.Gamma

        t0 = time.time()

        if self.stored_svd == 0:
            self.compute_svd(use_torch = use_torch)
            self.stored_svd = 1
        svd = self.svd

        # Computation of the threshold, similarly as numpy.linalg.matrix_rank:
        (m,n) = Gamma.shape
        eps = np.finfo(svd.dtype).eps
        tol =  svd.max() * max(m,n) * eps

        # Computation of the rank:
        r = np.sum(svd >= tol)

        t1 = time.time()
        self.rank = r

        if verbose != 0:
            print('Rank:', r)
            print('Computation time:', (t1 - t0)/60)
        
        return r