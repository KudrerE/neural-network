import pandas as pd
import numpy as np
from numpy import dtype
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 5000000000)
learning_rate=1e-5
i=0
# we have train and validation datas
def relu(Z):
    return np.maximum(0, Z)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=0)
def loss(x):
    return -np.log(x)

class Network:
    def __init__(self, url_data,url_test):
        self.w1 = np.random.rand(50,784) * 0.001
        self.w2 = np.random.rand(25, 50) * 0.001
        self.w3 = np.random.rand(10, 25) * 0.001
        self.data = pd.read_csv(url_data).to_numpy()
        self.real_test = pd.read_csv(url_test).to_numpy()
        # split for x and y
        number_of_rows = self.data.shape[0]
        random_indices = np.random.choice(number_of_rows, size=100, replace=False)
        self.sample = self.data[random_indices, :]
        data_y = self.sample[:int(self.sample.shape[0])][:, 0]
        data_x = self.sample[:int(self.sample.shape[0])][:, 1:]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42,stratify=data_y)
        self.x_train_transpose = self.x_train.T

    def show(self):
        print(self.w1, self.w2, self.w3, sep="\n")

    def shuffled_batch(self):
        number_of_rows = self.data.shape[0]
        random_indices = np.random.choice(number_of_rows, size=1000, replace=False)
        self.sample = self.data[random_indices, :]
        data_y = self.sample[:int(self.sample.shape[0])][:, 0]
        data_x = self.sample[:int(self.sample.shape[0])][:, 1:]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2,random_state=42, stratify=data_y)
        self.x_train_transpose = self.x_train.T

    def forward(self,x_t):
        self.z1=self.w1 @ x_t
        self.a1=relu(self.z1)
        self.z2 = self.w2 @ self.a1
        self.a2 = relu(self.z2)
        self.z3 = self.w3 @ self.a2
        self.a3 = softmax(self.z3)

    def backward(self):
        self.mem = np.array(self.a3, copy=True)
        self.mem[self.y_train, np.arange(0, self.a3.shape[1])] = -1 + self.a3[self.y_train, np.arange(0, self.a3.shape[1])]
        self.z3_grad = self.mem
        # WE FOUND W3 GRADİENT TO UPDATE
        # ---------------------------------------------------------------------------------------
        self.w3_gradient = self.z3_grad @ self.a2.transpose()
        self.a2_gradient = self.w3.transpose() @ self.z3_grad
        # ---------------------------------------------------------------------------------------
        self.z2_grad = np.array(self.a2_gradient, copy=True)
        self.z2_grad[self.z2 <= 0] = 0
        # WE FOUND W2 GRADİENT TO UPDATE
        # ---------------------------------------------------------------------------------------
        self.w2_gradient = self.z2_grad @ self.a1.transpose()
        self.a1_gradient = self.w2.transpose() @ self.z2_grad
        # ---------------------------------------------------------------------------------------
        self.z1_grad = np.zeros(self.a1_gradient.shape)
        z_psoitivity = self.z1 > 0
        self.z1_grad[z_psoitivity] = self.a1_gradient[z_psoitivity]
        # ---------------------------------------------------------------------------------------
        self.w1_gradient = self.z1_grad @ self.x_train

    def update(self):
        self.w1=self.w1-self.w1_gradient*learning_rate
        self.w2 = self.w2 - self.w2_gradient*learning_rate
        self.w3=self.w3-self.w3_gradient*learning_rate

    def loss(self):
        self.losses= np.sum(-np.log(self.a3[self.y_train, np.arange(0, self.a3.shape[1])]))
        print(self.losses)
    def validate(self):
        self.forward(self.x_test.T)
        u=self.a3.argmax(axis=0)
        self.accuracy=1-np.count_nonzero(self.y_test-u)/self.y_test.size
        print(self.accuracy)

    def submit (self):
        self.forward(self.real_test.T)
        u = my_network.a3.argmax(axis=0)
        cach=np.arange(1, my_network.a3.shape[1]+1)
        concavnew= np.hstack((cach,u))
        concavnew=np.reshape(concavnew,(2,28000))
        concavnew1 = pd.DataFrame(data = concavnew.T,columns = ["ImageId","Label"])
        print(concavnew1)
        concavnew1.to_csv('submission.csv',index=False)

    def train (self):
        self.shuffled_batch()
        self.forward(self.x_train_transpose)
        self.backward()
        self.update()
        self.loss()
        self.validate()

my_network =Network(r"C:\Users\kudre\Desktop\mavhine_try_data\train.csv", r"C:\Users\kudre\Desktop\mavhine_try_data\test.csv")

while my_network.losses<70 :
    my_network.train()
    if my_network.accuracy==1:
        i=i+1
    print(i)

my_network.submit()







