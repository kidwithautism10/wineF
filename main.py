import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


data = load_wine()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=999)
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy: {accuracy}")

plt.plot(mlp.loss_curve_, label="loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
