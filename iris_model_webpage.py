#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')


# ### Get the models from the pickle file

# In[2]:


# Load the models
log_model = pickle.load(open('log_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))


# In[4]:


html_temp = """ 
<div style = "background-color: #f06081; padding: 10px">
<h2 style = "color: white; text-align: center;">Iris Flower Classification
</div>
<div style = "background-color: white; padding: 5px">
<p style= "color: #7c4deb; text-align: center; font-family: Courier; font-size: 15px;"><i>This is an Iris flower classification app!</i></p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)


# In[5]:


image_path = 'iris.png'
image = Image.open(image_path)
st.image(image, use_column_width=True)


# #### Get the "class" of the iris species

# In[6]:


def classify(num):
    if (num < 0.5):
        setosa = Image.open('setosa.jpg')
        st.image(setosa, use_column_width=True)
        return 'Iris-setosa'
    elif (num < 1.5):
        versicolor = Image.open('versicolor.jpg')
        st.image(versicolor, use_column_width=True)
        return 'Iris-versicolor'
    else:
        verginica = Image.open('verginica.jpg')
        st.image(verginica, use_column_width=True)
        return 'Iris-virginica'


# #### Create a slider to give user options to select  a model

# In[7]:


activities = ['Logistic Regression', 'Support Vector Machine (SVM)']
option = st.sidebar.selectbox('Select your model of choice: ', activities)
st.subheader(option)


# #### Create sliders for the input features 

# In[8]:


def user_input_features():
    sl = st.slider('Sepal Length', 0.0, 8.0)
    sw = st.slider('Sepal Width', 0.0, 4.5)
    pl = st.slider('Petal Length', 0.0, 7.0)
    pw = st.slider('Petal Width', 0.0, 2.5)
    inputs = [[sl, sw, pl, pw]]
    
    features = pd.DataFrame(inputs, index=[0])
    features.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    return features


# In[9]:


df = user_input_features()
st.subheader('Input parameters')
st.write(df)


# ### Get the class prediction

# In[10]:


def prediction():
    if (st.button('Classification Result')):
        if (option == 'Logistic Regression'):
            st.success(classify(log_model.predict(df)))
        else:
            st.success(classify(svm_model.predict(df)))
prediction()


# ### Get the prediction probabilities for each class

# In[11]:


def prediction_probability():
    if (option == 'Logistic Regression'):
        prediction_prob = log_model.predict_proba(df)
    else:
        prediction_prob =svm_model.predict_proba(df)
    
    st.subheader('Prediction Probability')
    class_info = pd.DataFrame(prediction_prob, index=[0])
    class_info.columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    st.write(class_info)
prediction_probability()


# In[12]:


html_temp1 = """
    <div style = "background-color: #f27eac">
    <p style = "color: white; text-align: center;">Designed & Developed By: <b>Rajashri Deka</b></p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)


# In[ ]:




