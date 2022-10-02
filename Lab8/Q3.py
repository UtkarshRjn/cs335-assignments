
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn import svm


############################ Q3 ################################

def q3(x, w, b, conv_param):
    """
    A naive implementation of convolution.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    def conv_single_step(a_slice_prev, W, b):
    
      ### START CODE HERE ### (≈ 2 lines of code)
      # Element-wise product between a_slice and W. Do not add the bias yet.
      s = np.multiply(a_slice_prev,W)
      # Sum over all entries of the volume s.
      Z = np.sum(s)
      # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
      Z = Z + b.astype(float)
      ### END CODE HERE ###

      return Z

    N, C, H ,W  = x.shape
    F, C, HH, WW = w.shape
    padding, stride = conv_param['pad'], conv_param['stride']
    nH = int((H + 2* padding - HH) / stride) + 1
    nW = int((W + 2* padding - WW) / stride) + 1
    Z = np.zeros([N, F, nH, nW])
    X_pad = np.pad(x, ((0,0),(0,0), (padding,padding), (padding,padding)), 'constant', constant_values = (0,0))
    weight = w

    for i in range(N):
      x_pad = X_pad[i,:,:,:]
      for h in range(nH):
         for w in range(nW):                       # loop over horizontal axis of the output volume
                for c in range(F):                   # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride + HH
                    horiz_start = w*stride 
                    horiz_end = w*stride + WW
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    x_slice = x_pad[:,vert_start:vert_end,horiz_start:horiz_end]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, c, h, w] = conv_single_step(x_slice, weight[c, :, :, :], b[c])

    out = Z #Output of the convolution
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def gram(x):
  ######START: TO CODE########
  # Returns the gram matrix
  h, w, c = x.shape
  x = np.reshape(x, (h*w , c))
  return np.dot(x.T,x)
  ######END: TO CODE########



def relative_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))