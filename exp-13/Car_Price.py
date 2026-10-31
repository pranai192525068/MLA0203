
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample car dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Kilometers': [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000],
    'Price': [300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Year', 'Kilometers']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict car price
year = int(input("Enter car manufacturing year: "))
km = int(input("Enter kilometers driven: "))

prediction = model.predict([[year, km]])

print("Predicted Car Price: Rs.", round(prediction[0], 2))
