
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# load dtaa
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_ovr = LogisticRegression(max_iter=200)
model_ovr.fit(X_train, y_train)
y_pred_ovr = model_ovr.predict(X_test)

model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model_softmax.fit(X_train, y_train)
y_pred_softmax = model_softmax.predict(X_test)

# evaluateion func
def evaluate_model(y_true, y_pred, title):
    print(f"\n{title}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model(y_test, y_pred_ovr, "Logistic Regression (One-vs-Rest)")
evaluate_model(y_test, y_pred_softmax, "Logistic Regression (Softmax)")

input("Press Enter to exit...")
