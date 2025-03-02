import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from Code.LogisticRegression.logisticReg.logisticReg import LogisticRegression


def main():
    # training data

    train = pd.read_csv('../../Dataset/dataset_training_small.csv', sep=';')
    test = pd.read_csv('../../Dataset/dataset_test_small.csv', sep=';')

    vectorizer = CountVectorizer()

    X_train = train.drop('ai_generator', inplace=False, axis=1)
    vectorizer.fit(X_train['text'])
    X_train = vectorizer.transform(X_train['text']).toarray()

    y_train = train['ai_generator']
    y_train = y_train.to_numpy()

    X_test = test.drop('ai_generator', inplace=False, axis=1)
    X_test = vectorizer.transform(X_test['text']).toarray()

    y_test = test['ai_generator']
    y_test = y_test.to_numpy()

    n_features = X_train.shape[1]

    model = LogisticRegression(n_features)

    maxiter = 10
    maxfun = 10
    model.optim_model(X_train, y_train, maxiter, maxfun)

    print("Final cost:", model.cost_function(X_test, y_test))


if __name__ == '__main__':
    main()
