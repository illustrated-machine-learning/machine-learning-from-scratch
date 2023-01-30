import numpy as np


class LogisticRegression:

    def __init__(self,
                 max_iter = 10,
                 learning_rate = 0.1,
                 threshold = 0.5):

        self.losses = []
        self.accuracies = []
        self.weights = None
        self.bias = 0.0
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.threshold = threshold

    def fit(self,X,y):
        
        # initialize weights to 0
        self.weights = np.zeros(X.shape[1])

        # reshape y values
        y = y.reshape(y.shape[0], 1)

        for iter in range(self.max_iter):
            # calculate prediction
            y_pred = self.sigmoid(X,self.weights, self.bias)
            # calculate loss
            loss = self.get_loss(y_pred, y)
            # get gradients
            gradients_weights, gradient_bias = self.get_gradients(X, y_pred, y)
            # update weigths
            self.update_weights(gradients_weights, gradient_bias)

            # cast classes and get accuracy
            classes = [1 if probability > self.threshold else 0 for probability in y_pred]
            accuracy = self.get_accuracy(classes, y)

            self.accuracies.append(accuracy)
            self.losses.append(loss)


    def predict(self,X):
        ''' Calculate the relative probabilities through the sigmoid and 
        cast them to the selected class (0/1), based on the selected 
        threshold'''
        probabilities = self.sigmoid(X, self.weights, self.bias)
        return [1 if probability > self.threshold else 0 for probability in probabilities]

    
    def get_loss(self, y_pred, y):
        '''Binary cross entropy loss:
                average of the sums between the positive and negative losses'''
        return -np.mean( y * np.log(y_pred + 1e-10) + (1-y) * np.log(1 - y_pred + 1e-9))


    def get_gradients(self, X, y_pred, y):
        '''Derivative of the Binary Cross Entropy loss
        src: https://github.com/casperbh96/Logistic-Regression-From-Scratch/blob/main/src/logistic_regression/model.py'''
        
        gradients_weights = np.matmul(X.transpose(), y_pred - y)
        gradients_weights = np.array([np.mean(grad) for grad in gradients_weights])
        
        gradient_bias = np.mean(y_pred - y)

        return gradients_weights, gradient_bias

    
    def update_weights(self, gradients_weights, gradient_bias):
        ''' w = w - lr * gradients_weight
            b = b - lr * gradient_bias '''
        self.weights = self.weights - self.learning_rate * gradients_weights
        self.bias = self.bias - self.learning_rate * gradient_bias

    
    def get_accuracy(self, y_pred, y):
        return np.mean(y_pred == y)


    def sigmoid(self, x, weights, bias):
        ''' sigmoid(x,w,b) = 1 / (1 + e^(<x,w>+b)) '''
        z = np.dot(x,weights) + bias
        return 1.0 / (1 + np.exp(-z))



if __name__ == '__main__':

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.metrics import accuracy_score
    

    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative = 7,
        n_classes = 2, 
        random_state=42 
    )

    X_train, y_train = X[:700], y[:700]
    X_test, y_test = X[700:], y[700:]

    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print(f'Accuracy (M) = {accuracy_score(y_pred,y_test):.2f}')

    lr_scikit = LR()
    lr_scikit.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print(f'Accuracy (S) = {accuracy_score(y_pred,y_test):.2f}')


    