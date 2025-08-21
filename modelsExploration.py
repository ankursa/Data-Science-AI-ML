from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class nnModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None

    def createNN(self):
        self.model = Sequential()
        #self.model.add(Dense(64, input_dim=self.input_dim, activation='relu',use_bias=True))
        self.model.add(Dense(10, input_dim=self.input_dim, activation='relu',use_bias=False))
        self.model.add(Dense(self.output_dim,activation='linear',use_bias=False))
        return self.model

    def initializeNN(self):
        self.model = self.createNN()
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        return self.model

    def trainNN(self, X_train, y_train, batch_size, epochs):
        self.model = self.initializeNN()
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        return self.model

    def evaluateNN(self, X_test, y_test):
        self.model = self.initializeNN()
        loss, mse, mae = self.model.evaluate(X_test, y_test)
        return loss, mse, mae

if __name__ == "__main__":
    print("Experimenting with NN models!")

    nnModel = nnModel(input_dim=3, output_dim=1)
    nnModel.createNN()

    X_train_list = []
    y_train_list = []
    for i in range(500):
        hinge1 = random.randint(1,10)
        hinge2 = random.randint(50,100)
        y_tmp = 2*hinge1 + 0.05*hinge2 + 0.001*hinge1*hinge2
        x3 = random.randint(11,15)
        X_train_list.append([hinge1,hinge2,x3])
        y_train_list.append(y_tmp)
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    print(X_train, " : X_train : ",X_train.shape)
    print(y_train, " : y_train : ",y_train.shape)
    
    

    # Test data
    X_test_list = []
    y_test_list = []
    for i in range(15):
        hinge1 = random.randint(1,10)
        hinge2 = random.randint(50,100)
        y_tmp = 2*hinge1 + 0.05*hinge2 + hinge1*hinge2
        x3 = random.randint(11,25)
        X_test_list.append([hinge1,hinge2,x3])
        y_test_list.append(y_tmp)

    print("Model : ",nnModel)
    print("X_test : ",X_test_list)
    print("Y : ",y_test_list)

    X_train_scaled = MinMaxScaler().fit_transform(X_train_list)
    X_test_scaled = MinMaxScaler().fit_transform(X_test_list)

    y_train_scaled = MinMaxScaler().fit_transform(y_train_list)
    y_test_scaled = MinMaxScaler().fit_transform(y_test_list)

    #Scaled data
    print(X_train_scaled, " : X_train_scaled: ",X_train_scaled.shape)
    print(y_train_scaled, " : y_train_scaled : ",y_train_scaled.shape)

    exit(0)
    nnModel.trainNN(X_train= X_train_scaled, y_train=y_train_scaled, batch_size=50,epochs = 700)

    
    loss, mse, mae = nnModel.evaluateNN(X_test=np.array(X_test_list),y_test=np.array(y_test_list))
    print(f"Loss : {loss}, MSE : {mse}, MAE : {mae}")

    y_pred = nnModel.model.predict(np.array(X_test_list))
    print("Y_pred : ",y_pred)
    print("Y_actual : ",y_test_list)


