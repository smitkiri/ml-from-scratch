import numpy as np

def check_errors(y_true, y_pred):
    """
    Helper function which raises exceptions if needed.
    """
    if y_true.ndim != 1:
        raise IndexError("y_true should be 1 dimensional, shape should be (n,)")
                
    if y_pred.ndim != 1:
        raise IndexError("y_pred should be 1 dimensional, shape should be (n,)")
       
    if y_true.shape[0] != y_pred.shape[0]:
        raise IndexError("y_true and y_pred must have the same shape")
            
    if np.unique(y_true).shape[0] != 2 or np.unique(y_pred).shape[0] != 2:
        raise IndexError("Only binary classes supported.")
            
    return True
        
    
def accuracy(y_true, y_pred):
    """
    Returns the accuracy by comparing actual and predicted values.
    Parameters
    -----------
    y_true: Numpy 1d array
            Actual values of target variable
    y_pred: Numpy 1d array
            Predicted values of target variable
    """
    check_errors(y_true, y_pred)
       
    return np.sum(y_true != y_pred)/y_true.shape[0]
    
def precision(y_true, y_pred):
    """
    Returns the precision by comparing actual and predicted values.
    Assumes the class with higher number is positive class.
    Parameters
    -----------
    y_true: Numpy 1d array
            Actual values of target variable
    y_pred: Numpy 1d array
            Predicted values of target variable
    """
    check_errors(y_true, y_pred)
        
    p = np.unique[1]
    return np.sum(y_true[y_true == p] == y_pred[y_true == p])/np.sum(y_pred[y_true == p])
    
def recall(y_true, y_pred):
    """
    Returns the recall by comparing actual and predicted values.
    Assumes the class with higher number is positive class.
    Parameters
    -----------
    y_true: Numpy 1d array
            Actual values of target variable
    y_pred: Numpy 1d array
            Predicted values of target variable
    """
    check_errors(y_true, y_pred)
       
    p = np.unique[1]
    return np.sum(y_true[y_true == p] == y_pred[y_true == p])/np.sum(y_true[y_true == p])
    
def confusion_matrix(y_true, y_pred):
    """
    Returns the confusion matrix by comparing actual and predicted class.
    Assumes the class with higher number is positive class.
    Parameters
    -----------
    y_true: Numpy 1d array
            Actual values of target variable
    y_pred: Numpy 1d array
            Predicted values of target variable
    """
    check_errors(y_true, y_pred)
        
    n, p = np.unique(y_true)
    tp = np.sum(y_true[y_true == p] == y_pred[y_true == p])
    tn = np.sum(y_true[y_true == n] == y_pred[y_true == n])
    fp = np.sum(y_true[y_pred == p] != y_pred[y_pred == p])
    fn = np.sum(y_true[y_pred == n] != y_pred[y_pred == n])
        
    return np.array([tp, fp, fn, tn]).reshape(2,2)
    
def classification_report(y_true, y_pred):
    """
    Returns a printable string with confusion matrix, accuracy, precision, recall, F1 score
    Parameters
    -----------
    y_true: Numpy 1d array
            Actual values of target variable
    y_pred: Numpy 1d array
            Predicted values of target variable
    """
    check_errors(y_true, y_pred)
        
    n, p = np.unique(y_true)
    [tp, fp], [fn, tn] = confusion_matrix(y_true, y_pred)
    acc = (tp+tn)/(tp+fp+fn+tn)
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = (2*pre*rec)/(pre+rec)
        
    report = "Confusion Matrix:\n"
    report += " "+str(p)+"\t"+str(n)+"\n"
    report += str(p)+" "+str(tp)+"\t"+str(fp)+"\n"
    report += str(n)+" "+str(fn)+"\t"+str(tn)+"\n\n"
    report += "Accuracy: "+str(acc)+"\n"
    report += "Precision: "+str(pre)+"\n"
    report += "Recall: "+str(rec)+"\n"
    report += "F1 score: "+str(f1)
        
    return report