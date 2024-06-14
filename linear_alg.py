from sklearn.linear_model import LinearRegression

# Sample training data
X_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_train = [14, 32, 50]

# Create and train the model
model = LinearRegression().fit(X_train, y_train)

# Test data
X_test = [[10, 20, 30]]

# Predict the outcome for the test data
prediction = model.predict(X_test)

print("Predicted outcome:", prediction)