
import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = {
    'Income': [20000, 30000, 40000, 50000, 60000, 25000, 45000, 70000],
    'Credit': [600, 650, 700, 750, 800, 620, 720, 850],
    'Loan': [0, 0, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Income', 'Credit']]
y = df['Loan']

model = GaussianNB()
model.fit(X, y)

prediction = model.predict([[55000, 780]])

print("Bank Loan Prediction")

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
