#!/usr/bin/env python
# coding: utf-8

# In[21]:


import streamlit as st


# In[22]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
from textblob import TextBlob  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


# In[23]:


get_ipython().system('pip install textblob')


# In[24]:


def backpropagation_model(input_data):
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    learning_rate = 0.01
    max_epoch = 5000
    N = y_train.size
    input_size = X_train.shape[1]
    hidden_size = 2
    output_size = 3
    np.random.seed(10)
    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
    print(1/(1+np.exp(-x)))
    y_true_one_hot=np.eye(output_size)[y_true]
    y_true_reshaped = y_true_one_hot.reshape(y_pred.shape)
    error = ((y_pred - y_true_reshaped)**2).sum() / (2*y_pred.size)
    error= y_pred.argmax(axis=1) ==  y_true.argmax(axis=1)
    print(acc.mean())
    results = pd.DataFrame(columns=["mse", "accuracy"])
    for epoch in tqdm(range(max_epoch)):
        Z1 = np.dot(X_train, W1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(np.eye(output_size)[y_train], A2)
    new_row = pd.DataFrame({"mse": [mse], "accuracy": [acc]})
    E1 = A2 - np.eye(output_size)[y_train]
    dW1 = E1 * A2 * (1 - A2)
    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N
    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update
    results.mse.plot(title="Mean Squared Error")
    plt.show()
    results.accuracy.plot(title="Accuracy")
    plt.show()
    Z1 = np.dot(X_test, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    test_acc = accuracy(np.eye(output_size)[y_test], A2)
    print("Test accuracy: {}".format(test_acc))


# In[25]:


def rnn_model(input_data):
    dataset = pd.read_csv('https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection',sep='\t',names=['label','message'])
    print(dataset.head())
    print(dataset.groupby('label').describe())
    dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
    X = dataset['message'].values
    y = dataset['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    tokeniser = tf.keras.preprocessing.text.Tokenizer()
    tokeniser.fit_on_texts(X_train)
    encoded_train = tokeniser.texts_to_sequences(X_train)
    encoded_test = tokeniser.texts_to_sequences(X_test)
    print(encoded_train[0:2])
    max_length = 10
    padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
    padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
    print(padded_train[0:2])
    vocab_size = len(tokeniser.word_index)
    model=tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size,output_dim= 24, input_length=max_length),
        tf.keras.layers.SimpleRNN(24, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
      ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)
    model.fit(x=padded_train,y=y_train,epochs=50,validation_data=(padded_test, y_test),callbacks=[early_stop])
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {str(round(acc_sc,2)*100)}")
    return acc_sc
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, cmap="Blues", cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("C:/Users/Admin/Downloads/DL-ALGORITHMS-main/rnn/results/test.jpg")
    preds = (model.predict(padded_test) > 0.5).astype("int32")
    c_report(y_test, preds)
    plot_confusion_matrix(y_test, preds)
    model.save("RNN/results/model/spam_model")


# In[26]:


def lstm_model(input_data):
    tf.random.set_seed(7)
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save('lstm-code.h5')


# In[45]:


def perceptron_model(input_data):
    bias=0
    learning_rate=0.01
    max_epochs=100
    activation_function='step'
    x=0
    if activation_function=='step':
        return 1 if x>=0 else 0
    elif activation_function=='sigmoid':
        return 1 if (1 / (1 + np.exp(-x)))>=0.5 else 0
    elif self.activation_function == 'relu':
        return 1 if max(0,x)>=0.5 else 0
    n_features = X.shape[1]
    self.weights = np.random.randint(n_features, size=(n_features))
    for epoch in tqdm(range(self.max_epochs)):
        for i in range(len(X)):
            inputs = X[i]
            target = y[i]
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            prediction = self.activate(weighted_sum)
    print("Training Completed")
    predictions = []
    for i in range(len(X)):
        inputs = X[i]
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        prediction = self.activate(weighted_sum)
        predictions.append(prediction)
    return predictions


# In[28]:


def dnn_model(input_data):
    path = 'https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/iris.csv'
    df = pd.read_csv(path, header=None)
    X=df.values[:,:-1]
    y=df.values[:, -1]
    X = X.astype('float')
    y = LabelEncoder().fit_transform(y)
    print("Train | Test Dataset generation Started..")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=32)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    n_features = X_train.shape[1]
    output_class = 3
    model = Sequential()
    model.add(Dense(64, activation='sigmoid',input_shape=(n_features,)))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(output_class,activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Training Started.")
    history=model.fit(X_train, y_train, epochs=150, batch_size=20)
    loss, acc = model.evaluate(X_test, y_test)
    print("Training Finished.")
    print(f'Test Accuracy:{round(acc*100)}')
    row = [9.1,7.4,5.4,6.2]
    prediction = model.predict([row])
    print('Predicted: %s (class=%d)' % (prediction, argmax(prediction)))


# In[29]:


def cnn_model(input_data):
    (train_x, y_train), (test_x, y_test) = tf.keras.datasets.mnist.load_data()
    print("Size of training Data Loaded:\n")
    print('Train: X=%s, y=%s' % (train_x.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (test_x.shape, y_test.shape))
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
    test_x = test_x .reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))
    print('Train: X=%s, y=%s' % (train_x.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (test_x.shape, y_test.shape))
    train_x = train_x.astype('float')/255
    test_x = test_x.astype('float')/255
    shape = train_x.shape[1:]
    print("--------------------------------------\n")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape= shape))
    model.add(tf.keras.layers.MaxPool2D((2,2)))
    model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'])
    history = model.fit(train_x, y_train, epochs=5, batch_size = 128, validation_split = 0.2)
    print("Training Finished.\n")
    print("--------------------------------------\n")
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig("C:/Users/Admin/OneDrive/Desktop/deeplearning/DL-ALGORITHMS/CNN/mnist_sample/results/mnist_accuracy_plot.png")
    plt.clf()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig("C:/Users/Admin/OneDrive/Desktop/deeplearning/DL-ALGORITHMS/CNN/mnist_sample/results/mnist_loss_plot.png")
    loss,accuracy= model.evaluate(test_x, y_test)
    print(f'Accuracy: {round(accuracy*100,2)}')
    results = model.predict(test_x)
    results = argmax(results,axis = 1)
    results = pd.Series(results,name="Predicted Label")
    submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
    submission.to_csv("C:/Users/Admin/OneDrive/Desktop/deeplearning/DL-ALGORITHMS/CNN/mnist_sample/results/MNIST-CNN.csv",index=False)


# In[30]:


model_names=["Perceptron","BackPropagation","CNN","DNN","RNN","LSTM"]
task=["Image Classification","Sentiment Analysis"]
st.sidebar.title("Choose Model")
selected_model=st.sidebar.radio("Select Model:",model_names)
selected_task=st.sidebar.selectbox("Select Task:",task)


# In[52]:


if selected_model in ["Perceptron","CNN","DNN","RNN","LSTM"]:
    uploaded_image=st.file_uploader(r"C:\Users\Admin\OneDrive\Desktop\deeplearning\tumor_accuracy_plot.png",type=["jpg","jpeg","png"])


# In[53]:


if uploader is not None:
    image=Image.open(uploader)
    image=image.resize((224,224))
    image_array=keras.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0


# In[59]:


if selected_model == "Perceptron":
    prediction = perceptron_model(uploaded_image)
elif selected_model == "CNN":
    prediction = cnn_model(uploaded_image)
elif selected_model == "DNN":
    prediction = dnn_model(uploaded_image)
elif selected_model == "RNN":
    prediction = rnn_model(uploaded_image)
elif selected_model == "LSTM":
    prediction = lstm_model(uploaded_image)
elif selected_model=="Backpropagation":
    predication=backpropagation_model(uploaded_image)


# In[56]:


if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)


# In[57]:


st.subheader(f"{selected_model} Prediction:")
st.write(prediction)


# In[65]:


if selected_task == "Sentiment Analysis":
    text_input = st.text_area("Enter text for sentiment analysis:")
    analysis = TextBlob(text_input)
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Polarity: {analysis.sentiment.polarity}")
    st.write(f"Subjectivity: {analysis.sentiment.subjectivity}")

    if selected_model == "Perceptron":
        prediction = perceptron_model(uploaded_image)
    elif selected_model == "CNN":
        prediction = cnn_model(uploaded_image)
    elif selected_model == "DNN":
        prediction = dnn_model(uploaded_image)
    elif selected_model == "RNN":
        prediction = rnn_model(uploaded_image)
    elif selected_model == "LSTM":
        prediction = lstm_model(uploaded_image)


# In[64]:





# In[ ]:





# In[ ]:




