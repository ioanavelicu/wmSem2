from sklearn.datasets import load_wine
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")

precision = precision_score(y_test, y_pred, average='macro')
print(f"Test Precision: {precision:.2f}")

recall = recall_score(y_test, y_pred, average='macro')
print(f"Test Recall: {recall:.2f}")
