from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 0] 
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# threshold classification function
def classify(X, threshold):
    return (X > threshold).astype(int)

# accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

chosen_threshold = 5.0
y_train_pred = classify(X_train, chosen_threshold)
y_test_pred = classify(X_test, chosen_threshold)

train_acc = accuracy(y_train, y_train_pred)
test_acc = accuracy(y_test, y_test_pred)

print(f"Threshold used: {chosen_threshold}")
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Combine full dataset for ROC curve
X_all = np.concatenate([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
thresholds = np.linspace(min(X_all), max(X_all), 100)

tpr_list = []
fpr_list = []

for threshold in thresholds:
    y_pred = classify(X_all, threshold)
    tp = np.sum((y_all == 1) & (y_pred == 1))
    fp = np.sum((y_all == 0) & (y_pred == 1))
    fn = np.sum((y_all == 1) & (y_pred == 0))
    tn = np.sum((y_all == 0) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

#plot
plt.figure()
plt.plot(fpr_list, tpr_list, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()

input("Press Enter to exit...")
