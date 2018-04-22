from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weight=2 * random.random((3, 1))-1

    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    def predict(self, input):
        return self.__sigmoid(dot(input,self.synaptic_weight))

    def __sigmoid_derivative(self, x):
        return x*(1-x)
    
    def train(self, x, y, iterations):
        for it in range(iterations):
            output=self.predict(x)
            error=y-output
            adjustement = dot(x.T,error*self.__sigmoid_derivative(output))
            self.synaptic_weight+=adjustement


nn=NeuralNetwork()
print('Random weights !\n '+str(nn.synaptic_weight))
train_x=array([[0,0,1],[1,1,1],[1,0,1],[0,0,0]])
train_y=array([[1,1,1,0]]).T
nn.train(train_x,train_y,10000)
print('Final weight :'+str(nn.synaptic_weight))
print('Prediction :'+str(nn.predict([0,0,0])))