from classifiers import *




def main():
    X = load_breast_cancer()['data']
    y = load_breast_cancer()['target']
    noise = np.random.normal(0, 100, [X.shape[0], X.shape[1]])

    # X = X + noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train, X_test, y_test, epochs=200, lr=2)
    y_pred = logreg.predict(X_test)

    print("ROC AUC", roc_auc_score(y_test, logreg.predict(X_test)))
    print(classification_report(y_test, y_pred))

    naive = NaiveBayes()
    naive.fit(X_train, y_train)
    y_pred = naive.predict(X_test)

    print("ROC AUC NAIVE BAYES", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    #
    # print((np.dot(logreg.weights,logreg.weights)))
    # plt.plot(logreg.loss)
    # plt.plot(logreg.loss_test)
    #
    # plt.title('Logistic Regression Training', fontSize=15)
    # plt.xlabel('Epochs', fontSize=12)
    # plt.ylabel('Loss', fontSize=12)
    # plt.legend(["Train", "Test"])
    #
    # plt.show()


if __name__ == "__main__":
    main()