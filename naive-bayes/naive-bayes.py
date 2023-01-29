from collections import Counter

import numpy as np


class NaiveBayes:
    '''Implementation of the Naive Bayes classifier,
    given a categorical dataset'''

    def __init__(self):
        self.prior = {}
        self.likelihood = {}

    def fit(self, X, y):
        '''
        Bayes Theorem:
            P(C|X) = P(X|C) * P(C) / P(X)
        
        where:
        - P(C) is the prior belief (Nc / N),
        - P(X) is the same for everyone, therefore it is neglected,
        - P(X|C) is the probability of the current observation, given the class.
            P(X|C) = P(X1,...,Xk|C) = P(X1|C) * P(X2|C) * ... * P(Xk|C)
                P(Xi|C) = |Xic| / Nc
        '''

        X = np.asarray(X)

        # count the number of occurrences for each class, scaled by len(X)
        # Note: they are not yet scaled by N! 
        self.class_frequencies = Counter(y)
        self.prior = {k:v/len(X) for k,v in self.class_frequencies.items()}

        # init iteration parameters
        self.n_features = X.shape[1]
        self.classes = list(self.prior.keys())

        # calculate the likelihood of every feature, given the class => P(f|c)
        for idx in range(self.n_features): 
            self.likelihood[idx] = {}
            for c in self.classes:
                # First, filter out the elements having the current class as label,
                # with respect to the selected feature column (idx)
                # Then, count their frequencies of occurrence -> {element:cardinality}
                # Finally, scale this quantity for the number of occurrencies of the given class
                self.likelihood[idx][c] = { k:v/self.class_frequencies[c] for k,v in Counter(X[y == c,idx]).items()}
        


    def predict(self, X):
        '''Return the class having highest probability
        '''
        X = np.asarray(X)

        # N x C matrix with row probabilities
        # N is the number of elements
        # C is the number of classes
        self.probabilities = [self._get_class(x) for x in X]

        return [self.classes[np.argmax(probs)] for probs in self.probabilities]
        

    def _get_class(self, x):
        '''Return an array of length 'num_classes' containing the probability for
        any class, given a Row of our dataframe (x)
        '''
        try: 
            probabilities = [
                np.prod([self.likelihood[feature_idx][c][feature_value] for feature_idx,feature_value in enumerate(x)])
                    for c in list(self.classes)
            ]
            return probabilities
        except:
            return [0.0] * len(self.classes)
        




if __name__ == '__main__':

    from sklearn.metrics import accuracy_score

    import pandas as pd

    # src: https://www.kaggle.com/datasets/qizarafzaal/adult-dataset
    df = pd.read_csv('adult.csv', header=None, sep=',\s', engine='python')

    # assign column names
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df.columns = col_names

    # keep only categorical features
    keep = [
        'workclass', 'education','marital_status',
        'occupation', 'relationship', 'race', 
        'sex', 'native_country', 'income'
    ]  
    
    df = df[keep]

    y = df.income
    X = df.drop(columns=['income'])

    X_train, y_train = X[:25000], y[:25000]
    X_test, y_test = X[25000:], y[25000:]

    nb = NaiveBayes()
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)

    print(f'Accuracy = {np.sum(np.asarray(y_pred) == np.asarray(y_test)) / len(X_test) :.2f}')

