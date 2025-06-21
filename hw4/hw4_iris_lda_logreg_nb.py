
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# load data
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
n_classes = len(np.unique(y))

y_bin = label_binarize(y, classes=[0, 1, 2])

# models
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=200),
    "Naive Bayes": GaussianNB()
}

scores = {}
roc_curves = {}

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    scores[name] = (acc, f1)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_curves[name] = (fpr, tpr, roc_auc)

for name, (acc, f1) in scores.items():
    print(f"{name}: Accuracy = {acc:.2f}, F1 Score = {f1:.2f}")

# Plot
colors = ['blue', 'green', 'red']
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{name} - class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves - LDA vs Logistic Regression vs Naive Bayes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

input("Press Enter to exit...")
