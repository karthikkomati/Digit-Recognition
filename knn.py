# karthik komati
# this file is for an extension task

import sys

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

#main function to run the knn
def main(argv):
    data = pd.read_csv('intensities.csv', skiprows=[0], header=None)
    labels = pd.read_csv('categories.csv', skiprows=[0], header=None)

    #data.info()

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.40)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train.values.ravel())

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))

    print(cm)
    true_positive = cm[1][1]
    true_negative = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]

    print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
    print(f"Precision is {true_positive / (true_positive + false_positive)}")
    print(f"Recall is {true_positive / (true_positive + false_negative)}")
    fs = (true_positive) / (true_positive + (0.5 * (false_positive + false_negative)))
    print("fscore:", fs)

    mf = f1_score(y_test, y_pred, average='micro')
    print("Fscore micro", mf)

    # print("Fs sk learn:")
    # print(f1_score(y_test, y_pred))


    return

if __name__ == "__main__":
    main(sys.argv)


