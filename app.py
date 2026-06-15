import pandas as pd
df=pd.read_csv("data/house_price.csv")
print(df.head())

x=df[["Area","Bedrooms"]]
y=df["Price"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
print(mse)
predictions=model.predict([[1800,3]])
print(predictions)

