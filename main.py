import pandas as pd
import numpy as np
from numpy import log, dot, e
from numpy.random import rand
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class LogisticRegression:



    def sigmoid(self, z):
        return 1 / (1 + e ** (-z))

    def cost_function(self, X, y, weights):
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X) + self.lam*dot(weights,weights) / 2* len(X)

    def fit(self, X, y, Xtest, ytest, epochs=25, lr=0.05 , lam = 1):
        loss = []
        loss_test = []
        weights = rand(X.shape[1])
        self.lam = lam
        N = len(X)

        for _ in range(epochs):
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * (dot(X.T, y_hat - y) + lam * weights) / N
            # Saving Progress
            loss.append(self.cost_function(X, y, weights))
            loss_test.append(self.cost_function(Xtest, ytest, weights))


        self.weights = weights
        self.loss = loss
        self.loss_test = loss_test


    def predict(self, X):

        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.predict_proba(X)]

    def predict_proba(self, X):
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        return self.sigmoid(z)


# if __name__ == "__main__":
#
#     print('Humus!')

X = load_breast_cancer()['data']
y = load_breast_cancer()['target']
noise = np.random.normal(0, 100, [X.shape[0],X.shape[1]])


# X = X + noise

print(X.shape[0])
print(X.shape[1])
print(y.shape[0])




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train, X_test, y_test, epochs=200, lr=2)
y_pred = logreg.predict(X_test)


print("ROC AUC",roc_auc_score(y_test, logreg.predict_proba(X_test)))
print(classification_report(y_test, y_pred))

print((np.dot(logreg.weights,logreg.weights)))
plt.plot(logreg.loss)
plt.plot(logreg.loss_test)

plt.title('Logistic Regression Training', fontSize=15)
plt.xlabel('Epochs', fontSize=12)
plt.ylabel('Loss', fontSize=12)
plt.legend(["Train", "Test"])

plt.show()