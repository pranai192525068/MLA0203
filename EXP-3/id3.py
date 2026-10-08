import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Dataset
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain",
                "Rain", "Overcast", "Sunny", "Sunny", "Rain",
                "Sunny", "Overcast", "Overcast", "Rain"],

    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool",
                    "Cool", "Cool", "Mild", "Cool", "Mild",
                    "Mild", "Mild", "Hot", "Mild"],

    "Humidity": ["High", "High", "High", "High", "Normal",
                 "Normal", "Normal", "High", "Normal", "Normal",
                 "Normal", "High", "Normal", "High"],

    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak",
             "Strong", "Strong", "Weak", "Weak", "Weak",
             "Strong", "Strong", "Weak", "Strong"],

    "Play": ["No", "No", "Yes", "Yes", "Yes",
             "No", "Yes", "No", "Yes", "Yes",
             "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)

# Convert text data into numbers
X = pd.get_dummies(df[["Outlook", "Temperature", "Humidity", "Wind"]])
y = df["Play"]

# ID3 Decision Tree
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# New sample
new_data = pd.DataFrame([{
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}])

# Convert new sample
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict
result = model.predict(new_data)

print("ID3 Decision Tree")
print("-----------------")
print("New Sample: Sunny, Cool, High, Strong")
print("Classification:", result[0])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 10))

plot_tree(model,
          feature_names=X.columns,
          class_names=["No", "Yes"],
          filled=True,
          rounded=True)

plt.show()