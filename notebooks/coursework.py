import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dummy dataset
data = pd.DataFrame({
    'feature1': [1,2,3,4,5,6,7,8,9,10],
    'feature2': [5,3,6,9,2,7,1,8,4,0],
    'target':    [0,1,0,1,0,1,0,1,0,1]
})

X = data[['feature1', 'feature2']]
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy}")