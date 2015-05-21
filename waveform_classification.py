# -*- coding: utf-8 -*-
"""
@author: olgamac
"""

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import preprocessing, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cluster
from sklearn.cross_validation import  LeaveOneOut
from sklearn import cross_validation


from scipy.stats import sem
import numpy as np

MYPATH_TRAIN = "/Users/olgamac/Documents/Python_data/SpaceX/project/submission/files_train/"
MYPATH_PREDICT = "/Users/olgamac/Documents/Python_data/SpaceX/project/submission/files_predict/"

TARGET = np.array([1,1,1,1,0,0,0,0,0,0,0,0])

class Signal():
        
    def load(self, mypath, name):
        '''
        Loads files from the porovided location
        '''
        self.name = name
        data = np.genfromtxt(mypath + name, delimiter=',')
        return data[:,-1]  
        
    def plot(self, input_data):
        '''
        Plots the waveform
        '''
        plt.figure()
        plt.plot(input_data)
        plt.show()
    
    def get_file_name(self):
        '''
        Returnes the name of the file, that's been loaded
        '''
        return self.name
    
    def rolling_window(self, input_data, window):
        ''' Takes time series as an imput_data and a window parameter
            to perform rolling-window filtering
            If provided data has more than one column, only the last column is used
        '''
        shape = input_data.shape[:-1] + (input_data.shape[-1] - window + 1, window)
        strides = input_data.strides + (input_data.strides[-1],)
        return np.mean(np.lib.stride_tricks.as_strided(input_data, shape=shape, strides=strides),-1)
        
    def diff(self, input_data):
        '''
        Returnes first derivative of the time series input_data
        '''
        return np.diff(input_data)
        
    def peak_count(self, input_data):
        '''
        Counts peaks of the time series. 
        The time series input data is a first derivative
        '''
        #tresholding the signal to remove the noise and sink the baseline below zero
        data_temp = input_data - 0.01
        peak_counter = 0    
        for index in range(len(data_temp)-1):
            if data_temp[index]*data_temp[index+1] < 0:
                peak_counter += 1
        return int(peak_counter/2)
        
    def integral(self, input_data):
        pass
    
    def interval(self, input_data):
        pass
    
    def scaling(self, input_data):
        '''
        scaling the entire feature space to the mean 0
        takes the input_data as an argument, returnes scaled data back
        '''
        scaler = preprocessing.StandardScaler().fit(input_data)
        return scaler.transform(input_data)
        
    def PCA_transform(self,input_data, n_components):
        '''
        Principal Component Analysis over the input_data at the given n_component
        plots the eignevalues of the principal components
        returnes the transformed array with the number of columnes equal to n_components
        '''
        estimator = PCA(n_components = n_components)
        X_transform = estimator.fit_transform(input_data)
        mean = np.mean(X_transform.T,axis=1)
        demeaned = X_transform-mean
        evals, evecs = np.linalg.eig(np.cov(demeaned.T))
        plt.figure()
        plt.plot(evals)
        plt.title('Eigenvalues of the PCA components')
        plt.xlabel('Prinicipal component #')
        plt.ylabel('Eigenvalue')
        return X_transform

class ML():
    
    def measure_performance(self, X, y, clf, show_accuracy = True, show_classification_report = True, show_confusion_matrix = True):
        '''
        Takes the data, list of corresponding labeles and a classifyer
        returnes the accuracy of prediction, classification report and confusion matrix
        '''
        print "\nMeasure algorithm performance"
        y_pred = clf.predict(X)
        if show_accuracy:
            print "Accuracy:{0:.3f} ".format(metrics.accuracy_score(y,y_pred)),"\n"
        if show_classification_report:
            print "Classification report"
            print metrics.classification_report(y,y_pred),"\n"
        if show_confusion_matrix:
            print "Confusion matrix:"
            print metrics.confusion_matrix(y,y_pred),"\n"
            
    def classify_Learner(self, input_data, metrics = True):
        '''
        Splits the input_data into training and testing sets, 
        classifyes using GradientBoosting algorithm
        prints the performance measured against the portion of the data reserved for testing
        returnes the classifyer
        '''
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(input_data, TARGET, test_size = 0.25, random_state = 10)
        clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=10, random_state=0).fit(X_train, y_train)
        if metrics:
            self.measure_performance(X_test, y_test, clf)
        return clf
        
    def classify_real_examples(self, input_data, metrics = True):  
        '''
        Trains the classifyer on the input_data as whole
        using GradientBoosting algorithm
        returnes classifyer
        '''
        clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=10, random_state=0).fit(input_data, TARGET)
        if metrics:
            self.measure_performance(input_data, TARGET, clf)
        return clf
        
    def predict(self, input_data, input_test): 
        '''
        Trains classifyer based on input_data using GradientBoosting
        returnes the predicted labeles on input_test data
        '''
        prediction = []
        print "\nData classification"
        for index in range(len(input_test)):
            y_pred = self.classify_real_examples(input_data, metrics = False).predict(input_test[index])
            prediction.append(y_pred)
            print "file#", index, "   predicted class ", y_pred
        return prediction
    
    def LOO(self, input_data, target, clf):
        '''
        Leave One Out cross-validation method
        Takes data and classifyer as arguments
        returnes mean and sem of the predicted score over all iterations
        '''
        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(input_data, target, test_size = 0.25, random_state = 33)
        X_train = input_data
        y_train = target
        
        loo = LeaveOneOut(X_train[:].shape[0])
       
        scores = []
        for train_index, test_index in loo:
            X_train_cv, X_test_cv = X_train[train_index],X_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            clf = clf.fit(X_train_cv, y_train_cv)
            y_pred = clf.predict(X_test_cv)
            scores.append(metrics.accuracy_score(y_test_cv, y_pred))
        print ("Leave One Out mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
        return [np.mean(scores), sem(scores)]

########## Helpers functions ##########################################
def preprocessing_data(mypath):
    '''
    Helper function to manipulate the data to get it ready to apply Machin Learning.
    load, filter, count peaks, concotinate
    takes a path to the folder where the files are saved as an argument
    returns an array of features to use for training/testing/prediction
    '''
    signal = Signal()
    #loading waveforms
    onlyfiles =  [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]  
    onlyfiles = sorted(onlyfiles)
    data = np.array([signal.load(mypath, file) for file in onlyfiles])

    #filtering noise out
    data_filtered = np.array([signal.rolling_window(waveform,15) for waveform in data])

    #taking first derivative
    data_diff = np.array([signal.diff(waveform) for waveform in data_filtered])

    #filtering noise from first derivative
    data_diff_filtered = np.array([signal.rolling_window(waveform,10) for waveform in data_diff])

    #counting peaks of the waveform using first derivative
    peaks_number = np.array([signal.peak_count(waveform) for waveform in data_diff_filtered])[...,None]
    
    #concatenates the reduced data and information on peak number
    all_data = signal.scaling(np.concatenate((signal.PCA_transform(data, 6), peaks_number), 1))

    return all_data

def write_to_file(mypath, prediction):
    '''
    Helper function to generate txt file with the results of classification 
    prints the names of the files stored in mypath and predicted classification label for each file
    '''
    onlyfiles =  [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] 
    with open('Predictive_results','w') as file_out:
        for index in range(len(onlyfiles)):
            new_line = onlyfiles[index] + ',' + str(prediction[index]) + '\n'
            file_out.write(new_line)  
    
###################  SIGNAL PROCESSING  ############################   
#data_train is 12 waveformes provided upon wich the learner is constructed
data_train = preprocessing_data(MYPATH_TRAIN)

#data_predict is the data to be classified by the Learner
data_predict = preprocessing_data(MYPATH_PREDICT)

#####################  CLASSIFICATION ################################  
learner = ML()

learner.classify_Learner(data_train)

#unsupervised - showed number of predicted classes
aff = cluster.AffinityPropagation()
aff.fit(data_train)
print "# of predicted clusters ", aff.cluster_centers_indices_.shape 

clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=10, random_state=0)
learner.LOO(data_train,TARGET, clf)


####################  PREDICTION  ####################################
prediction = learner.predict(data_train, data_predict)
write_to_file(MYPATH_PREDICT, prediction)

