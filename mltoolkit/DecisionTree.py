import numpy as np
import copy

class DecisionTree:
    """
    A decision tree that supports binary target variables.
    """
    def __init__(self, max_depth = 1):
        """
        Parameters
        -----------
        max_depth: int
                   The maximum depth of the decision tree
        """
        self.max_depth = max_depth
        self.root = {}
    
    def node_entropy(self, subset, y1 = 0, y2 = 1, weights = None):
        """
        Calculates the entropy for one split node.
        The target variable should have only two classes.
        Returns entropy and number of elements in the subset.
        Parameters
        -----------
        subset:  Numpy 1d array
                 The subset of target variable created after applying the splitting criteria.
        
        weights: Numpy 1d array
                 Sample weights for each observation
        """
        n = subset.shape[0]
        if n == 0:
            return np.inf, n
        
        if weights is not None:
            if n != weights.shape[0]:
                raise IndexError('Weights does not have same dimensions. Expected length '+str(n)+', got length '+str(len(weights)))
            
            n0 = np.sum(weights[subset == y1])
            n1 = np.sum(weights[subset == y2])
        
        else:
            n0 = np.sum(subset == y1)
            n1 = np.sum(subset == y2)
        
        eps = np.finfo(float).eps #A small value to add to the log function to avoid getting an error
        return -(n0/(n0+n1)+eps)*np.log2(n0/(n0+n1)+eps) - (n1/(n0+n1)+eps)*np.log2(n1/(n0+n1)+eps), n
    
    
    def get_entropy(self, y, criteria, weights = None):
        """
        Returns entropy after a split.
        Parameters
        -----------
        y:        Numpy 1d array  
                  The target variable. Should only contain binary values
        
        criteria: Numpy 1d array
                  A boolean list indicating the indexes to select after the splitting criteria is matched.  
        
        weights:  Numpy 1d array or List type object
                  Sample weights for each observation
        """
        n = y.shape[0]
        
        if n == 0:
            return np.inf
        try:
            y1, y2 = np.unique(y)
        except:
            raise ValueError('Target does not have binary classes.')
            
        if weights is not None:
            weights = np.array(weights)
            entropy_true, n_true = self.node_entropy(y[criteria], y1, y2, weights[criteria])
            entropy_false, n_false = self.node_entropy(y[~criteria], y1, y2, weights[~criteria])
        
        else:
            entropy_true, n_true = self.node_entropy(y[criteria], y1, y2)
            entropy_false, n_false = self.node_entropy(y[~criteria], y1, y2)
    
        return (n_true/n)*entropy_true + (n_false/n)*entropy_false
    
    
    def best_split(self, col, y, weights = None):
        """
        Returns the best splitting value and its entropy.
        Parameters
        -----------
        col:     Numpy 1d array
                 A column from a Pandas-DataFrame.
                 Should be numeric and continious.
        
        y:       Numpy 1d array
                 The target column to predict.
        
        weights: Numpy 1d array or List type object
                 Sample weights for each observation
        """
        min_entropy = np.inf
        cutoff = None
        
        for val in set(col):
            criteria = col < val
            entropy = self.get_entropy(y, criteria, weights)
            if entropy < min_entropy:
                min_entropy = entropy
                cutoff = val
        return cutoff, min_entropy
    
    
    def best_attribute(self, features, target_col, weights = None):
        """
        Returns the best splitting attribute, its entropy and cutoff value for that attribute.
        Parameters
        -----------
        features:   Numpy 2d array
                    The features used for prediction.
        
        target_col: Numpy 1d array
                    The column to predict.
        
        weights:    Numpy 1d array or List type object
                    Sample weights for each observation
        """
        attribute = None
        min_entropy = np.inf
        best_cutoff = None
        
        for col, val in enumerate(features.T):
            cutoff, entropy = self.best_split(val, target_col, weights)
            if entropy == 0:
                return col, entropy, cutoff
            elif entropy < min_entropy:
                attribute = col
                min_entropy = entropy
                best_cutoff = cutoff
        
        return attribute, min_entropy, best_cutoff

    
    def _buildTree(self, x, y, weights = None, depth = 0, parent_node = {}):
        """
        Recursiverly builds the decision tree
        Parameters
        -----------
        x:           Numpy 2d array
                     Training features
        
        y:           Numpy 1d array
                     Training labels. Should only have binary classes
        
        weights:     Numpy 1d array or List type object
                     Sample weights for each observation
        
        depth:       int 
                     Current depth of tree
        
        parent_node: Dictionary
                     Parent Node of current node 
        """
        if depth > self.max_depth: #Max depth is reached
            return {}
        
        elif len(y) == 0: #No data in this group
            return {}
        
        elif all(val == y[0] for val in y): #All same labels
            return {'val': y[0]}
        
        else:
            attribute, entropy, cutoff = self.best_attribute(x, y, weights)
            
            #Data for left hand side
            left = x[:, attribute] < cutoff
            x_left = x[left]
            y_left = y[left]
            weights_left = None
            if weights is not None:
                weights_left = weights[left]
            
            #Data for right hand side
            right = x[:, attribute] >= cutoff
            x_right = x[right]
            y_right = y[right]
            weights_right = None
            if weights is not None:
                weights_right = weights[right]
            
            #Add data to node
            value, count = np.unique(y, return_counts = True)
            pred = value[np.argmax(count)]
            parent_node = {'attribute': attribute, 'cutoff': cutoff, 'val': pred}
            
            #Generate left hand side tree
            parent_node['left'] = self._buildTree(x_left, y_left, weights_left, depth+1, {})
            
            #Generate right hand side tree
            parent_node['right'] = self._buildTree(x_right, y_right, weights_right, depth+1, {})
            
            return parent_node
            
    
    def fit(self, x, y, weights = None):
        """
        Trains a decision tree classifier
        Parameters
        -----------
        x:       Numpy 2d array or Pandas Dataframe object
                 Training features
           
        y:       Numpy 1d array or Pandas Series object
                 Training labels
           
        weights: Numpy 1d array or List type object
                 Sample weights for each observation
        """
        
        x = np.array(x)
        y = np.array(y)
        
        self.root = self._buildTree(x, y, weights)
        return self
        
    def _get_predictions(self, row):
        """
        Predicts the target variable for single observation
        Parameters
        -----------
        row: 1d Numpy array
             Features for a single observation
        """
        node = copy.copy(self.root)
        
        while node.get('attribute'):
            if row[node['attribute']] < node['cutoff']:
                if not node['left']:
                    return node.get('val')
                node = node['left']
            else:
                if not node['right']:
                    return node.get('val')
                node = node['right']
        else:
            return node.get('val')
    
    def predict(self, x):
        """
        Predicts the target variable
        Parameters
        -----------
        x: Pandas DataFrame object or 2d Numpy array
           Features required to predict target variable.
        """
        
        x = np.array(x)
        pred = np.zeros(x.shape[0])
        
        for index, row in enumerate(x):
            pred[index] = self._get_predictions(row)
            
            
        return pred