# Naive Bayes Classifier

Naive Bayes Classifier is a `probabilistic machine learning algorithm` based on `Bayes' theorem`:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

where, 

* $P(A|B)$ is the posterior probability of hypothesis $A$ given evidence $B$.
* $P(B|A)$ is the likelihood of the evidence $B$ given hypothesis $A$.
* $P(A)$ is the prior probability of hypothesis $A$.
* $P(B)$ is the prior probability of evidence $B$.


The previous equation states that the probability of a hypothesis given some observed evidence is proportional to the prior probability of the hypothesis and the likelihood of the evidence given the hypothesis. 

In the context of classification, the `hypothesis` is the class label, and the evidence is the `features` of the input data. This algorith makes a `naive` assumption that the features are independent.


---

📍 The full implementation is available [here](./naive-bayes.py)!

--- 

### Demo

`Import` the requested libraries and load the dataset. For the sake of simplicty we'll filter out just 8 categorical features. 

```python
from sklearn.metrics import accuracy_score
import pandas as pd

# src: https://www.kaggle.com/datasets/qizarafzaal/adult-dataset
df = pd.read_csv('adult.csv', header=None, sep=',\s', engine='python')

# assign column names
col_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
df.columns = col_names

# keep only categorical features
keep = ['workclass', 'education','marital_status','occupation', 'relationship', 'race', 'sex', 'native_country', 'income']  
df = df[keep]
```

`Split` the dataset.

```python
y = df.income
X = df.drop(columns=['income'])

X_train, y_train = X[:25000], y[:25000]
X_test, y_test = X[25000:], y[25000:]
```

`Evaluate` the model.

```python
nb = NaiveBayes()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

print(f'Accuracy = {np.sum(np.asarray(y_pred) == np.asarray(y_test)) / len(X_test) :.2f}')
```

```
Accuracy = 0.75
```