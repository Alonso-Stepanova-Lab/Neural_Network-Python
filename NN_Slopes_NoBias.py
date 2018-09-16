from datetime import datetime
startTime = datetime.now()
import numpy as np
import matplotlib.pyplot as plt

#define the neural network
number_of_neurons = 10
number_of_inputs = 1
number_of_output_neurons = 1
number_of_training_sets =  1000

speed=500
increment_for_slope=0.01



X=np.random.rand(number_of_training_sets,number_of_inputs)
X=X/np.max(X)
#Y=np.sum(X,axis=1)
Y=np.sin(X*12)
#Y=3*X
Y=1/(1+np.exp(-Y))




#here we create a class corresponding to the main functions of the neural_network

class Neural_Network(object):
    def __init__(self):
        # here we will inicialize the weigths
        self.Win = np.random.randn(number_of_inputs, number_of_neurons)
        self.Wout = np.random.randn(number_of_neurons, number_of_output_neurons)


    def yHat_calculator(self,X):
        #this will produce an estiamte of the output based on the inputs and the weigths
        self.act1 = np.dot(X,self.Win)
        self.sig_act1 = self.sigmoid(self.act1)
        self.act2 = np.dot(self.sig_act1,self.Wout)
        self.yHat = self.sigmoid(self.act2)
        return self.yHat


    def cost_calculator(self,Y):
        self.yHat=self.yHat_calculator(X)
        self.cost = (sum(((Y-self.yHat)**2)/2))/number_of_training_sets 
        return self.cost
        #print(self.cost)


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))


class Trainer(object):
    def __init__(self,N):
        self.N=N
        self.yHat=self.N.yHat_calculator(X)
        self.Jold=self.N.cost_calculator(Y)
        self.Winori=np.zeros(self.N.Win.shape)
        self.Woutori = np.zeros(self.N.Wout.shape)


    def train(self):
        for q in range(1000):
        #while sum(abs(Y-self.N.yHat)) >0.002:
            for p1 in range(number_of_inputs):
                for p2 in range(number_of_neurons):
                    self.Winori[p1,p2]=self.N.Win[p1,p2]
                    self.N.Win[p1,p2]=self.N.Win[p1,p2] + increment_for_slope
                    self.N.yHat_calculator(X)
                    self.Jnew = self.N.cost_calculator(Y)
                    self.slope = (self.Jnew - self.Jold)

                    self.N.Win[p1,p2] = self.Winori[p1,p2] - (self.slope)*speed
                    self.N.yHat_calculator(X)
                    self.Jold = self.N.cost_calculator(Y)

                    self.Woutori[p2,p1]=self.N.Wout[p2,p1]
                    self.N.Wout[p2,p1]=self.N.Wout[p2,p1]+increment_for_slope
                    self.N.yHat_calculator(X)
                    self.Jnew = self.N.cost_calculator(Y)
                    self.slope=(self.Jnew-self.Jold)
                    self.N.Wout[p2,p1] = self.Woutori[p2,p1] - (self.slope) * speed
                    self.N.yHat_calculator(X)
                    self.Jold = self.N.cost_calculator(Y)

                self.cost=self.N.cost_calculator(Y)
                plt.plot(q,self.cost,'r.')
            
        plt.show()
        self.q=q

            #print(self.N.yHat)






NN=Neural_Network()
#NN.cost_calculator(Y,X)
T=Trainer(NN)
T.train()

for w in range(1000):
    NewX=w/1000
    #YNew=3*NewX
    YNew=np.sin(NewX*12)
    #YNew=1/(1+np.exp(-YNew))
    #plt.plot(NewX,NN.yHat_calculator(NewX),'c.')
    YY= np.log(NN.yHat_calculator(NewX)/(1-NN.yHat_calculator(NewX)))
    plt.plot(NewX,YY,'c.')
    plt.plot(NewX,YNew,'r.')
plt.show()




#NewX=0.5
#NewY=np.sin(NewX*6)
#NewY=1/(1+np.exp(-NewY))


print (' Time elapsed=',datetime.now() - startTime)
#print(' Expected result=',YNew,'\n','Result from the NN=',NN.yHat_calculator(NewX))
print(' Expected result=',YNew,'\n','Result from the NN=',YY)
print(' Number_of_neurons=',number_of_neurons,'\n','Number_of_inputs=',number_of_inputs,'\n',
'Number_of_output_neurons=',number_of_output_neurons,'\n','Number_of_training_sets=',number_of_training_sets,'\n','Speed=',speed)
print(' Cost=',T.cost)