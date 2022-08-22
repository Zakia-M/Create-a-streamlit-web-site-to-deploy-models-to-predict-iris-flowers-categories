import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

st.write('''
#   Simple app for Iris flowers prediction
This application predicts the category of Iris flowers
''')

st.sidebar.header("Input parametres")

def user_input():
    # create sliders 
    sepal_length=st.sidebar.slider('Sepal Length',4.3,7.9,4.0) #(min, max, default)
    sepal_width=st.sidebar.slider('Sepal Width',2.0,4.4,3.0)
    petal_length=st.sidebar.slider('Petal Length',1.0,6.9,2.0)
    petal_width=st.sidebar.slider('Petal Width',0.1,2.5,1.0)
    # put the inputs on a dictionary
    data={
    'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width
    }
    #put the params on a dataframe (index=0 coz 1 input=one line)
    flower_parametres=pd.DataFrame(data,index=[0])
    return flower_parametres

df=user_input()

st.subheader('We want to find the category of this flower')
# show df
st.write(df)

iris=datasets.load_iris()
model1=RandomForestClassifier()
model2=LogisticRegression()
model3=GaussianNB()

model1.fit(iris.data,iris.target)
model2.fit(iris.data,iris.target)
model3.fit(iris.data,iris.target)

prediction1=model1.predict(df)
prediction2=model2.predict(df)
prediction3=model3.predict(df)

pred={
    'Random Forest ':iris.target_names[prediction1][0],
    'Logistic Regression':iris.target_names[prediction2][0],
    'Gaussian Naive Bais':iris.target_names[prediction3][0],
    }
predicted_flower_category=pd.DataFrame(pred,index=[0])

st.subheader("The category of the iris flower is:")
st.write(predicted_flower_category)
