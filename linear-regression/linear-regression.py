from sklearn.metrics import r2_score
import numpy as np


class SimpleLinearRegression:

    def __init__(self):
        '''y = a + bX'''
        self.a = 0
        self.b = 0
    
    def fit(self,X,y):
        '''N.B. X is a 1d vector'''
        self.b = (np.sum((X - np.mean(X)) * (y - np.mean(y)))) / (np.sum((X-np.mean(X))**2))
        self.a = np.mean(y) - (self.b * np.mean(X))

    def predict(self, x):
        return self.a + self.b * x


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn as sns

    X = np.linspace(0,10,10)

    a, b  = 3, -2
    y = b * X + a + 0.7 * np.random.randn(X.shape[0])

    lr = SimpleLinearRegression()
    lr.fit(X,y)

    y_fit = lr.a + lr.b * X

    sns.scatterplot(x=X,y=y, color="#77DD77")
    sns.lineplot(x=X,y=y_fit, color="#FF6961")
    
    plt.xlabel("X")
    plt.xlabel("y")
    plt.legend(["Ground truth","Fitted line"])
    plt.title("Linear Regression")
    
    plt.show()
