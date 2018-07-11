import numpy as np
import cPickle
import os

import logging
logger = logging.getLogger(__name__)

class XORWrapper(object):
  def __init__(self, *args, **kwargs):
    '''
    Creates a XOR Wrapper instance.
    '''

    self.T = 4

    self.build_dir = os.path.join(".", "build")
    self.filename = os.path.join(self.build_dir,"XORDataSet.pickle")

    if not os.path.exists(self.build_dir):
      os.mkdir(self.build_dir)
    #end if

    self.X_train = np.array([[0,0],
                             [0,1],
                             [1,0],
                             [1,1]])
    self.Y_train = np.array([0,1,1,0]).reshape(-1,1)

    self.X_val, self.Y_val = self.X_train, self.Y_train
    self.X_test, self.Y_test = self.X_train, self.Y_train
      
    self.N_test = self.Y_test.shape[0]
    self.N_train = self.Y_train.shape[0]
    self.N_val = self.Y_val.shape[0]

    return super(XORWrapper, self).__init__(*args, **kwargs)
  def __len__(self):
    ''' Returns the length of the trainings data set '''
    return self.N_train

  def get_data_shape(self):
    ''' Returns the shape of one data sample '''
    return self.X_test.shape[1:]

  def sort_data_set(self, X,Y):
    ''' Call this function with a data set (X,Y) to get a sorted version back (label is index of list) '''
    assert Y.ndim == 2 and Y.shape[1] == 1, "Only 1-dimensional target values can be sorted"
    
    X_sorted = [[],[]]

    for x,y in zip(X,Y):
      X_sorted[y[0]].append(x)

    return map(np.array, X_sorted)
  def one_hot_targets(self, Y):
    '''
    Converts the arrays of class labels 0,1 to a one hot encoded representation
    '''
    assert Y.ndim == 2 and Y.shape[1] == 1, "Only 1-dimensional target values can be one-hot encoded"

    # Number of class labels is the largest number plus one in the training label set
    # +1 required because class labels start at zero
    classes = Y.max() + 1
    samples = Y.shape[0]

    Y_onehot = np.zeros((samples, classes))
    Y_onehot[np.arange(samples),Y[:,0]] = 1
    
    return Y_onehot

  def plot_data_set(self,run, X_sorted, label_markes={0:"o", 1:"s"}, title=None):
    ''' Plots the given sorted data set'''
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig = plt.figure()
    fig.canvas.set_window_title(run)

    for i, Xi in enumerate(X_sorted):
      if Xi.shape == (0,): continue
      plt.scatter(Xi[:,0],Xi[:,1], marker = label_markes[i], label = str(i))
          
    if title is not None:
      plt.title(title)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    
    return
  def plot_training_data_set(self):
    ''' Plots the current training data set'''
    X_sorted = self.sort_data_set(self.X_train, self.Y_train)
    return self.plot_data_set("XOR training data set", X_sorted, title = "XOR training data set")
  def plot_classifier(self, run, classifier):
    ''' Plots the labels given by the classifier for the test data set'''
   
    predictions = np.apply_along_axis(classifier, 1, self.X_test).argmax(axis = 1)
    X_sorted = self.sort_data_set(self.X_test, predictions)

    return self.plot_data_set(run, X_sorted, title = "Classifier Output")
  def plot_solution(self, run):
    ''' Plots the correct labels for the test data set'''
    X_sorted = self.sort_data_set(self.X_test, self.Y_test)
    return self.plot_data_set(run, X_sorted, title = "Correct Classification")

if __name__ == "__main__":

  wrapper = XORWrapper()

  wrapper.plot_training_data_set()

  print "All done"
