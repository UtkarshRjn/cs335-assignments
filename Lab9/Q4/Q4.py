import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os



############################ Q3 ################################

def GaussianFilter(x, w,stride):
    """
    A naive implementation of gradient filter convolution.
    The input consists of N data points,height H and
    width W. We convolve each input with F different filters and has height HH and width WW.
    Input:
    - x: Input data of shape (N, H, W)
    - w: Filter weights of shape (F, HH, WW)
    - stride: The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    Return:
   - out: Output data, of shape (N, F, H, W)
    """
    ##Note:if the mean value from a filter is float,perform ceil operation i.e.,29.2--->30
    #################### Enter Your Code Here

    def conv_single_step(a_slice_prev, W):
    
      ### START CODE HERE ### (≈ 2 lines of code)
      # Element-wise product between a_slice and W. Do not add the bias yet.
      s = np.multiply(a_slice_prev,W)
      # Sum over all entries of the volume s.
      Z = np.ceil(np.sum(s)/np.sum(W))
      ### END CODE HERE ###

      return Z

    N, H ,W  = x.shape
    F, HH, WW = w.shape
    padding = 1
    nH = int((H + 2* padding - HH) / stride) + 1
    nW = int((W + 2* padding - WW) / stride) + 1
    Z = np.zeros([N, F, nH, nW])
    X_pad = np.pad(x, ((0,0), (padding,padding), (padding,padding)), 'constant', constant_values = (0,0))
    weight = w

    for i in range(N):
      x_pad = X_pad[i,:,:]
      for h in range(nH):
         for w in range(nW):                       # loop over horizontal axis of the output volume
                for c in range(F):                   # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride + HH
                    horiz_start = w*stride 
                    horiz_end = w*stride + WW
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    x_slice = x_pad[vert_start:vert_end,horiz_start:horiz_end]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, c, h, w] = conv_single_step(x_slice, weight[c, :, :])
                    if(h == 0 or h== nH-1 or w == 0 or w == nW - 1): Z[i,c,h,w] = x_slice[1,1]
    
    out = Z

    return out

x_shape = (1, 6,6)
w_shape = (1,3,3)
x = np.array([[15,20,25,25,15,10],[20,15,50,30,20,15],[20,50,55,60,30,20],[20,15,65,30,15,30],[15,20,30,20,25,30],[20,25,15,20,10,15]]).reshape(x_shape)
w = np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]]).reshape(w_shape)
stride=1
out = GaussianFilter(x, w, stride)
# correct out=array([[[[15, 20, 25, 25, 15, 10],
#          [20, 29, 38, 35, 24, 15],
#          [20, 36, 48, 43, 29, 20],
#          [20, 31, 42, 37, 27, 30],
#          [15, 24, 29, 25, 22, 30],
#          [20, 25, 15, 20, 10, 15]]]])