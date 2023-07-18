import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # (D, C)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_trains = X.shape[0] # N
  num_classes = W.shape[1] # C

  for i in range(num_trains):
    score = np.dot(X[i], W)

    # trick 在进入softmax前先减去最大值，避免指数函数溢出
    score -= np.max(score)
    # ⬆️

    softmax = np.exp(score) / np.sum(np.exp(score)) # （1， C）
    current_loss = -np.log(softmax[y[i]])
    loss += current_loss

    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += X[i] * (softmax[j] - 1)
      elif j != y[i]:
        dW[:, j] += X[i] * softmax[j]

  loss /= num_trains
  dW /= num_trains

  loss += reg * np.sum(W*W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_trains = X.shape[0]
  score = np.dot(X, W) # (N, C)
  score -= np.max(score, axis=1).reshape(num_trains, 1)
  softmax = np.exp(score) / np.sum(np.exp(score), axis=1).reshape(num_trains, 1)
  loss = np.sum(-np.log(softmax[np.arange(num_trains), y]))
  softmax[np.arange(num_trains), y] -= 1
  dW = np.dot(X.T, softmax)

  loss /= num_trains
  dW /= num_trains

  loss += reg * np.sum(W*W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

