#Single layer perceptron given the following inputs from csv file:
"""
b  x1	x2	y
1  0.5	0.5	0
1  0.25	1	0
1  0	1	0
1  2.5	3.45	0
1  1.5	2	0
1  0.5	4	1
1  1.5	6	1
1  1.45	2.6	1
1  0	1.1	1
1  0.5	1.6	1

#bias is concatenated in later
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#define functions

#training weights
def predict(matrix,weight):
    
    
    #weighted sum
    totalsum = 0

    for matrix, weight in zip(matrix,weight): 
        totalsum+=matrix*weight

    #activation function - threshold 
    thres = 0 #the threshold to be how far above or below the line the value is (scaled with regard to weights)
    return 1 if totalsum > thres else 0
               
    
#determine accuracy
def accuracy(matrix,weight):
    NoCor = 0 #number of correct guesses
    preds = [] #define array for predicted classification
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1],weight) #get prediction for i excluding class
        preds.append(pred) #append in prediction array
        if matrix[i][-1]==pred: #if prediction = correct result
            NoCor+=1
            #print(preds[i],matrix[i][-1])
    return NoCor/float(len(matrix))

#plot epoch
def plot(matrix,weight):
    x1_0=[] #class 0 x values
    x2_0=[] #class 0 y values
    x1_1=[] #class 1 x values
    x2_1=[] #class 1 y values
    
    for i in range(len(matrix)):
        if matrix[i][-1]==1:
            x1_1.append(matrix[i][1])
            x2_1.append(matrix[i][2])
        if matrix[i][-1]==0:
            x1_0.append(matrix[i][1])
            x2_0.append(matrix[i][2])
    #print(x1_1,x2_1)
    
    xdata = np.linspace(-0.5,3,10)
    ydata = -(weight[1]*xdata+weight[0])/weight[2]
    
    fig = plt.figure(figsize = (8,8))
    plt.scatter(x1_1,x2_1,color='b',label="Class 1")
    plt.scatter(x1_0,x2_0,color='r',label="Class 0")
    plt.plot(xdata,ydata)
    #plt.ylim(0,7)
    #plt.xlim(0,10)
    #print("y = ",weight[1],"x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

#predict
def train_weights(matrix,weight,NoEpoch,Lr):
    #iterate number of times given
    for epoch in range(NoEpoch):
        #calculate current accuracy
        cur_acc = accuracy(matrix,weight)
        #print("\nEpoch %d \nWeights: "%epoch,weight)
        if cur_acc == 1:
            plot(matrix,weight)
            print("100% Accuracy!!!")
            print("\nEpoch %d \nWeights: "%epoch,weight)
            print("\nlearning Rate: ",Lr)
            break
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weight)
            #print(prediction, weight)
            error = matrix[i][-1]-prediction
            #print("\nEr: ",error)
            for j in range(len(weight)):
                #print(matrix[i][j])
                #change in weight = weight + (l_rate*error*differential wrt weight)
                weight[j]=weight[j] + (Lr*error*matrix[i][j])
        #print(cur_acc)
        #plot(matrix,weight)

def main():
    df = pd.read_csv("trainingdata.csv")
    #print(df)

    #convert database into matrix
    matrix = df.values
    #print(matrix[0])
    
    NoEpoch = 30 #number of iterations
    Lr = 10 #learning rate
    
    #for simplicity
    x1 = np.zeros(len(matrix)) #input x1
    x2 = np.zeros(len(matrix)) #input x2
    y = np.zeros(len(matrix)) #desired output
    
    
    #initialise weights w1,w2,b
    w1 = np.round(random.uniform(0, 1),3) #bias weight
    w2 = np.round(random.uniform(0,1),3) #w1
    w3 = np.round(random.uniform(0,1),3) #w2
    b = np.full(len(matrix),1)#np.round(random.uniform(0, 1),3)) #add bias
    #matrix = np.append(matrix[:][3],b)
    b = np.tile(b[np.newaxis,0], (matrix.shape[0],1)) #allow b to become 2D, matrix.shape to tile b to matrix format
    #print(b,"\n")
    #print(matrix.shape[0],1)
    matrix = np.concatenate((b,matrix), axis=1) #add bias
    
    weights = [w1,w2,w3]
    print(matrix)
    #print(weights)
    #print(matrix[:][1])
    #for i in range (len(matrix)):
    #    print(matrix[i][1])
    #predict(matrix,weights)
    train_weights(matrix,weights,NoEpoch,Lr)
    #print(x1)
    #print(x2)
    #return 0

if __name__ =="__main__":
    main()
