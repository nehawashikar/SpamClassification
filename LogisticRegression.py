import numpy as np
import csv
import matplotlib.pyplot as plt 

'''
Goal: classify emails as "ham" (not spam --> 0) or "spam"(1)
Each row is an email with different features labeled is spam or ham
"Label" column classifies email as spam or ham. Other labels are input features used for prediction
Run with 'python3 LogisticRegression.py'
'''
class LogisticRegression:
    #Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. 
    def __init__(self):
        #the learning rate
        self.__rate = 0.01

        #the weights to learn (float)
        self.__weights = []

        #the number of iterations
        self.__ITERATIONS = 200    


    '''
    Implement the sigmoid function
    maps any real-valued number into a value between 0 and 1
    '''
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    '''
    Helper function for prediction 
    Takes a test instance as input and outputs the probability of the label being 1 **/
    This function should call sigmoid()
    '''
    def predict_prob(self, test):
        sum = np.dot(test, self.__weights)
        return self.sigmoid(sum)


    '''
    The prediction function
    Takes a test instance as input and outputs the predicted label **/
    This function should call Helper function **/
    '''
    def predict_label(self, test):
        prob = self.predict_prob(test)
        if prob >= 0.5:
            return 1
        else:
            return 0


    '''
    This function takes a test set as input, call the predict function to predict a label for it, **/
    and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix
    '''
    def get_accuracy(self, test_set):
        data = self.read_input(test_set)
        features = data[:, :-1].astype(float) #all features
        labels = data[:, -1].astype(float) #all labels

        #get array of predictions for every line in feature 
        predictions = np.array([self.predict_label(line) for line in features])

        #compare with labels for confusion matrix 
        TP = np.sum((predictions == 1) & (labels == 1))
        FP = np.sum((predictions == 1) & (labels == 0))
        TN = np.sum((predictions == 0) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))
        print(f"Confusion Matrix:\nTP: {TP}   FP: {FP}\nFN: {FN}   TN:{TN}")

        #calc and print accuracy
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        print("\nAccuracy: ", accuracy)

        #calc and print Precision P for positive class (spam)
        precision = TP /(TP+FP)
        print("\nPrecision (Spam):", precision)

        #calc and print recall R for positive class (spam)
        recall = TP/(TP+FN)
        print("Recall (Spam):", recall)

        #calc and print F1 for positive class (spam)
        f1 = 2 * ((precision*recall)/(precision+recall))
        print("F1 Score (Spam):", f1)

        #calc and print Precision P for negative class (ham)
        precision_neg = TN /(TN+FN)
        print("\nPrecision (Ham):", precision_neg)

        #calc and print recall R for negative class (ham)
        recall_neg = TN/(TN+FP)
        print("Recall (Ham):", recall_neg)

        #calc and print F1 for negative class (ham)
        f1_neg = 2 * ((precision_neg*recall_neg)/(precision_neg+recall_neg))
        print("F1 Score (Ham):", f1_neg)


    '''
    Train the Logistic Regression in a function using Stochastic Gradient Descent **/
    Also compute the log-loss in this function
    '''
    def train(self, training_set):
        data = self.read_input(training_set)
        features = data[:, :-1].astype(float) #all features
        labels = data[:, -1].astype(float) #all labels       
        # Initialize weights to zeros
        self.__weights = np.zeros(features.shape[1])

        log_loss_values = []

        #SGD randomly picks one data point at each iteration to reduce the computation drastically
        for iteration in range(self.__ITERATIONS):
            #pick data point
            #compute gradient
            #update weights using gradient (move to minimally reduce cost)
            #repeat until number of iterations is reached

            for index in range(len(features)):
                feature = features[index] #get the feature 
                label = labels[index] #get the label 

                predicton_prob = self.predict_prob(feature) #find the probability of the feature
                error = label - predicton_prob #find the error of the feature
                gradient = feature * error #get the gradient

                self.__weights += self.__rate * gradient
            
            #get the prediction for every feature
            predictions = self.predict_prob(features)
            log_loss = np.mean(-labels * np.log(predictions) - ((1 - labels) * np.log(1 - predictions)))
            log_loss_values.append(log_loss)
            print(f"Iteration {iteration + 1}/{self.__ITERATIONS}, Log Loss: {log_loss}")
        
        #calc total log loss
        print("\nTotal loss: ", np.mean(log_loss_values))

        #plot log loss values 
        plt.plot(log_loss_values)
        plt.title('Log Loss vs. Training Iteration')
        plt.xlabel('Training Iteration')
        plt.ylabel('Log Loss')
        plt.show()

    #Function to read the input dataset
    def read_input(self, f):
        with open(f, 'r', newline='') as csv_file:
            #iterate over lines, convert to list
            csv_iter = csv.reader(csv_file)
            next(csv_iter, None)
            csv_list = list(csv_iter)
            #covert to array and float for linear regression
            data = np.array(csv_list)
        return data
        

# /** main Function **/
print("test")
lr = LogisticRegression()
lr.train("train-1.csv")

print("\n------------------------------Training Set------------------------------")
lr.get_accuracy("train-1.csv")

print("\n------------------------------Test Set------------------------------")
lr.get_accuracy("test-1.csv")

print("\n------------------------------------------------------------")
data = lr.read_input("train-1.csv")
labels = data[:, -1].astype(float)
print("Number of Positives in Training Set: ", int(np.sum(labels)))
print("Number of Negatives in Training Set: ", int(len(labels) - np.sum(labels)))