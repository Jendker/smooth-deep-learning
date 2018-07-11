import numpy as np
import scipy
import scipy.io

import cPickle
import os

import logging
logger = logging.getLogger(__name__)

class FourRegionWrapper(object):
  def __init__(self, use_file=False, T = 1000, *args, **kwargs):
    '''
    Creates a FourRegion Wrapper instance.

    If use_file is true, the wrapper first tries to load existing data from the disk. 
    If this fails a new data set is created and stored.

    => use_file = True allows for reproducible experiments without worrying about setting the seed of numpy
    '''

    self.x_range = (-4,4)
    self.y_range = (-4,4)

    self.radii = [1.0,2.0,3.0]

    self.T = T

    self.build_dir = os.path.join(".", "build")
    self.filename = os.path.join(self.build_dir,"FourRegionDataSet.pickle")

    if not os.path.exists(self.build_dir):
      os.mkdir(self.build_dir)
    #end if

    try:

      if not use_file:
        raise IOError("The existing file has been ignored, reason: use_file = False")

      # Code to load matlab matrices for comparing results
      #import scipy.io
      #foo = scipy.io.loadmat("four_region_v2.mat")
      #
      #pidx = [int(a) - 1 for a in "535    90   112   136   676   493   189   492   147    55".split("   ")]
      #
      #self.X_train = foo["X_input"].T[pidx,:]
      #self.Y_train = foo["Y_output"].T[pidx,:]
      #
      #self.X_test = self.X_train.copy()
      #self.Y_test = self.Y_train.copy()
      #self.X_val = self.X_train.copy()
      #self.Y_val = self.Y_train.copy()

      with open(self.filename, "rb") as the_file:
        result = cPickle.load(the_file)
      #end with
      
      self.X_train, self.Y_train_raw, self.X_val, self.Y_val_raw, self.X_test, self.Y_test_raw = result

      logger.info("Restored data set from file.")

    except IOError as e:

      self.X_train, self.Y_train_raw = self.create_data_set(self.T)
      self.X_val, self.Y_val_raw = self.create_data_set(self.T / 2)
      self.X_test, self.Y_test_raw = self.create_data_set(self.T / 2)

      with open(self.filename, "wb") as the_file:
        cPickle.dump((self.X_train, self.Y_train_raw, self.X_val, self.Y_val_raw, self.X_test, self.Y_test_raw), the_file, cPickle.HIGHEST_PROTOCOL)
      #end with

      # Create the matlab version for debugging
      # scipy.io.savemat('data_4r_1k_random.mat', dict(X_in=self.X_train, Y_out=self.Y_train))
      
      logger.info("Created a new data set.")

    #end if
      
    self.Y_train = self.one_hot_targets(self.Y_train_raw)
    self.Y_val = self.one_hot_targets(self.Y_val_raw)
    self.Y_test = self.one_hot_targets(self.Y_test_raw)
        
    self.N_test = self.Y_test.shape[0]
    self.N_train = self.Y_train.shape[0]
    self.N_val = self.Y_val.shape[0]

    return super(FourRegionWrapper, self).__init__(*args, **kwargs)
  def __len__(self):
    ''' Returns the length of the trainings data set '''
    return self.N_train

  def get_data_shape(self):
    ''' Returns the shape of one data sample '''
    return self.X_test.shape[1:]
  
  def region_1(self, x):
    ''' Returns true, if the sample x lies within this region '''
    return x[1] > 0.0 and np.linalg.norm(x) > self.radii[2]
  def region_2(self, x):
    ''' Returns true, if the sample x lies within this region '''
    return x[1] <= 0.0 and np.linalg.norm(x) > self.radii[2]
  def region_3(self, x):
    ''' Returns true, if the sample x lies within this region '''
    r = np.linalg.norm(x)

    if x[0] <= 0.0:
      return self.radii[1] <= r <= self.radii[2]
    else:
      return self.radii[0] <= r <= self.radii[1]
  def region_4(self, x):
    ''' Returns true, if the sample x lies within this region '''
    r = np.linalg.norm(x)

    if x[0] <= 0.0:
      return 0.0 <= r <= self.radii[1]
    else:
      return 0.0 <= r <= self.radii[0] or self.radii[1] <= r <= self.radii[2]
  def get_label(self, x):
    ''' Returns the label for the sample x by checking the regions '''
    if self.region_1(x):
      return 0
    elif self.region_2(x):
      return 1
    elif self.region_3(x):
      return 2
    elif self.region_4(x):
      return 3
    else:
      raise ValueError("The sample {} can not be labeld".format(x))

  def create_data_set(self, T):
    ''' Call this function to create T samples and their labels '''
    
    X = np.array([np.random.uniform(*self.x_range, size = (T,)),
                  np.random.uniform(*self.y_range, size = (T,))]).T

    Y = np.array(map(self.get_label, X)).reshape(-1,1)

    return X,Y
  def sort_data_set(self, X,Y):
    ''' Call this function with a data set (X,Y) to get a sorted version back (label is index of list) '''
    assert Y.ndim == 2 and Y.shape[1] == 1, "Only 1-dimensional target values can be sorted"
    
    classes = np.max(Y) + 1

    X_sorted = [[] for _ in xrange(classes)]

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
  def plot_data_set(self,run, X_sorted, label_markes={0:"o", 1:"s", 2:"h", 3:"1"}, title=None):
    ''' Plots the given sorted data set'''
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig = plt.figure()
    fig.canvas.set_window_title(run)

    for i, Xi in enumerate(X_sorted):
      if Xi.shape == (0,): continue
      plt.scatter(Xi[:,0],Xi[:,1], marker = label_markes[i], label = str(i))

    ax = fig.gca()
    ax.set_axisbelow(True)
    ax.add_patch(patches.Rectangle((self.x_range[0],self.y_range[0]), self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0],fill = False))
    ax.add_patch(patches.Circle((0,0), self.radii[1],fill = False))
    ax.add_patch(patches.Circle((0,0), self.radii[2],fill = False))
    ax.add_patch(patches.Arc((0,0), 2.0 * self.radii[0],2.0 * self.radii[0],90.0,180.0,fill = False))
    ax.add_patch(patches.Polygon(np.array([(0.0,self.radii[2]),(0.0,self.radii[0])]),closed = True,edgecolor = 'k'))
    ax.add_patch(patches.Polygon(np.array([(0.0,-self.radii[0]),(0.0,-self.radii[2])]),closed = True,edgecolor = 'k'))
    ax.add_patch(patches.Polygon(np.array([(self.x_range[0],0.0),(-self.radii[2],0.0)]),closed = True,edgecolor = 'k'))
    ax.add_patch(patches.Polygon(np.array([(self.x_range[1],0.0),(self.radii[2],0.0)]),closed = True,edgecolor = 'k'))

    if title is not None:
      plt.title(title)

    plt.xlabel("x")
    plt.xlim(self.x_range)
    plt.ylim(self.y_range)
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    
    return
  def plot_training_data_set(self):
    ''' Plots the current training data set'''
    X_sorted = self.sort_data_set(self.X_train, self.Y_train_raw)
    return self.plot_data_set("Four Region training data set", X_sorted)
  def plot_classifier(self, run, classifier):
    ''' Plots the labels given by the classifier for the test data set'''
   
    predictions = np.apply_along_axis(classifier, 1, self.X_test).argmax(axis = 1).reshape(-1,1)
    X_sorted = self.sort_data_set(self.X_test, predictions)

    return self.plot_data_set(run, X_sorted, title = "Classifier Output")

  def plot_solution(self, run):
    ''' Plots the correct labels for the test data set'''
    X_sorted = self.sort_data_set(self.X_test, self.Y_test_raw)
    return self.plot_data_set(run, X_sorted, title = "Correct Classification")

if __name__ == "__main__":

  wrapper = FourRegionWrapper()

  wrapper.plot_training_data_set()

  from matplotlib import pyplot as plt
  plt.show()

  print "All done"
