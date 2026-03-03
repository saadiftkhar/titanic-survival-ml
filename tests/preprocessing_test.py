# tests/preprocessing_test.py
import pandas as pd

# Load the dataset directly from URL, same as train.py
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Preprocessing (copy of your train.py steps)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna('S')
data['Sex'] = data['Sex'].map({'male':0, 'female':1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Test: No missing values in Age or Embarked
assert data['Age'].isnull().sum() == 0, "Error: Missing values remain in 'Age' column!"
assert 'Embarked_Q' in data.columns and 'Embarked_S' in data.columns, "Error: Embarked one-hot columns missing!"

# Test: Sex column properly encoded
unique_sex = set(data['Sex'].unique())
assert unique_sex.issubset({0, 1}), f"Error: Unexpected values in 'Sex' column: {unique_sex}"

# Test: Selected features exist
required_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
for feat in required_features:
    assert feat in data.columns, f"Error: Feature '{feat}' missing in processed data!"

print("All preprocessing tests passed!")