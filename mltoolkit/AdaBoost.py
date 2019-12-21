import numpy as np
import timeit
import copy

class AdaBoost:
    def __init__(self, base_classifier, num_iterations = 5):
        """
        Parameters
        -----------
        base_classifier: object with fit() function to train a model
                         Base learner
        
        num_iterations:  int
                         Number of iterations to perform
        """
        self.classifier = base_classifier
        self.iterations = num_iterations
        self.weights = []
        self.hypothesis = []
    
    def fit(self, x, y, verbose = 0):
        """
        Trains AdaBoost Classifier.
        Parameters
        -----------
        x:          Pandas DataFrame object or Numpy 2d array
                    Training data
        
        y:          Pandas Series object or Numpy 1d array
                    Training labels, should only have binary classes
        
        verbose:    int
                    set 1 to print the current running iteration,
                    otherwise does not print anything
        """
        x = np.array(x)
        y = np.array(y)
        n = y.shape[0]
        w = np.ones(n)/n

        for t in range(self.iterations):
            start = timeit.default_timer()
            if verbose == 1:
                print("Iteration ", t+1, end='')
            
            classifier = copy.copy(self.classifier) #Creating a copy of the bease classifier
            classifier.fit(x, y, w) #Training the base model

            pred = classifier.predict(x) #Predicting values
            
            incorrect = (pred != y).astype(float) #Getting the incorrect predictions' indices
            
            error = np.sum(np.dot(w, incorrect)) #Calculating weighted error
            
            beta = (1/2)*np.log((1-error)/error) #Calculating model weights
            
            w = w * np.exp(np.where(incorrect, beta, -beta)) #Updating weights
            w = w/np.sum(w)

            self.hypothesis.append(classifier)
            self.weights.append(beta.copy())
            
            stop = timeit.default_timer()
            if verbose == 1:
                print("\t\tTime: "+ str(round(stop-start, 2))+"s")
        
        return self
    
    def predict(self, x):
        """
        Predicts the target variable
        Parameters
        -----------
        x: Pandas DataFrame object
           Features required to predict target variable.
        """
        if not self.hypothesis:
            raise ReferenceError('Model not trained yet.')
        
        pred = np.zeros(len(x))
        
        for (classifier, beta) in zip(self.hypothesis, self.weights):
            pred += beta * classifier.predict(x)
                   
        return np.sign(pred)