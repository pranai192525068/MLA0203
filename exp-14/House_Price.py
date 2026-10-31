
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample house dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4],
    'Price': [2000000, 2400000, 3000000, 3600000,
              4000000, 4400000, 5000000]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Get user input
area = int(input("Enter house area in sq.ft: "))
bedrooms = int(input("Enter number of bedrooms: "))

# Predict price
prediction = model.predict([[area, bedrooms]])

print("Predicted House Price: Rs.", round(prediction[0], 2))
