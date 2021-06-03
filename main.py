import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error
#from app import *
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Stock Market")
st.write("""
LSTM Meachine Learning
""")

dataset=st.sidebar.selectbox("Select DataSet",("APPLE","HPQ"))
def get_dataset(dataset):
    if dataset == 'APPLE':
        data=pd.read_csv('AAPL.csv')
        st.write("Stock Data")
        st.write(data)

    else :
        data = pd.read_csv('HPQ.csv')
        st.write("Stock Data")
        st.write(data)
    return data
data = get_dataset(dataset)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()
def get_xtrain(data):
    data1 = data.filter(['Close'])
    dataset1 = data1.values
    training_data_len = math.ceil(len(dataset1) * .8)
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scalar.fit_transform(dataset1)
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    #        # if i <= 61:
    #            # st.write(x_train)
    #             #st.write(y_train)
               # st.write()

    return x_train,y_train,data


x_train,y_train,dataset2=get_xtrain(data)
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
#x_train.shape

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')

def trainmodel(x_train,y_train,model,ep):



    model.fit(x_train ,y_train, batch_size=1,epochs=ep)

    return model
ep=int(st.number_input("Enter Epochs"))
c=st.button("Test And Train Model")
if c:

    model=trainmodel(x_train,y_train,model,ep)

dataset2 = dataset2.filter(['Close'])
dataset3 = dataset2.values
training_data_len1 = math.ceil(len(dataset3) * .8)
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data1 = scalar.fit_transform(dataset3)
test_data=scaled_data1[training_data_len1 - 60: , :]
#create x_test and y_test

x_test=[]
y_test=dataset3[training_data_len1:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test=np.array(x_test)

x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1] , 1 ))
predictions=model.predict(x_test)
predictions=scalar.inverse_transform(predictions)

st.write("Root Mean Square Error")
rmse =np.sqrt(np.mean(((predictions- y_test)**2)))
st.write(rmse)


st.write("Mean Absolute Error")
MAe= mean_absolute_error(y_test, predictions)
st.write(MAe)

train=dataset2[:training_data_len1]
valid=dataset2[training_data_len1:]
valid['Predictions']=predictions


def show(data, training_data_len, predictions):
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # showing the data
    plt.figure(figsize=(8, 4))
    plt.title('Model')
    plt.xlabel('data', fontsize=8)
    plt.ylabel('closed price $', fontsize=8)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])

    plt.legend(['Train', 'Test', 'Predictions'], loc='upper left')
    st.pyplot()

def showtr(data,training_data_len):
    train = data[:training_data_len]
    valid = data[training_data_len:]
    # showing the data
    plt.figure(figsize=(8, 4))
    plt.title('Model')
    plt.xlabel('data', fontsize=8)
    plt.ylabel('closed price $', fontsize=8)
    plt.plot(train['Close'])
    plt.plot(valid['Close'])

    plt.legend(['Train', 'Test'], loc='upper left')
    st.pyplot()



st.write("Test and train")
showtr(data,training_data_len1)
st.write("Model Accuracy Plot")
show(data,training_data_len1,predictions)

st.write("Model Accuracy Data")
st.write(valid)


