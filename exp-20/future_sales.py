
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8],
    'Sales': [100, 150, 200, 250, 300, 350, 400, 450]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

future = pd.DataFrame({'Month': [9, 10, 11, 12]})

prediction = model.predict(future)

print("Future Sales Prediction")

for i in range(len(future)):
    print("Month", future['Month'][i],
          "Sales:", prediction[i])
