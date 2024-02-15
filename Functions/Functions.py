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
        l = len(layers)     # Number of layers
        if use_logits == 1:     # If 1, do not take into account the output activation function
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

        # Number of identifiable parameters of the model (maximum rank)
        self.Npar_identif = self.Npar - np.sum(self.arch) + self.arch[0] + self.arch[L]


        # Function computing the gradients with respect to the parameters
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
        
        
    def get_differential(self, X):
        
        self.stored_svd = 0
        self.stored_norm = 0
        self.stored_frob = 0
        Nsample = X.shape[0]


        jacob = self.jacob


        arch = self.arch
        L = self.number_layers

        # Creation of the matrix that will contain the jacobian
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

        self.Gamma = Gamma

        return Gamma


    def compute_svd(self, use_torch = 1):

        Gamma = self.Gamma
        if self.stored_svd == 0:        # Avoid unnecessary computations of the SVD
            if use_torch == 1:
                Gamma_torch = torch.from_numpy(Gamma)
                svd = torch.linalg.svdvals(Gamma_torch)
                svd = svd.numpy()
            else:
                svd = np.linalg.svd(Gamma, compute_uv=False)  
            self.svd = svd      # Store the computed spectrum
            self.stored_svd = 1
        
        svd = self.svd
        
        return svd
    
    
  
    def compute_rank(self, use_torch = 1):
        
        Gamma = self.Gamma

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

        self.rank = r

        return r