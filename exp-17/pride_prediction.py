
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Create dataset
data = {
    'RAM': [2, 3, 4, 6, 8, 12, 4, 8],
    'Battery': [2000, 2500, 3000, 4000, 4500, 5000, 3500, 4800],
    'Storage': [16, 32, 32, 64, 128, 256, 64, 128],
    'Price': [0, 0, 1, 1, 2, 3, 1, 2]
}

df = pd.DataFrame(data)

X = df[['RAM', 'Battery', 'Storage']]
y = df['Price']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Get mobile details
ram = int(input("Enter RAM: "))
battery = int(input("Enter Battery: "))
storage = int(input("Enter Storage: "))

# Predict price
prediction = model.predict([[ram, battery, storage]])

print("Predicted Mobile Price Range:", prediction[0])
print("0=Low, 1=Medium, 2=High, 3=Very High")
