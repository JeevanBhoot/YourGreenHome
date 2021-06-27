import numpy as np
import scipy.special
import csv

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih1 = np.random.normal(0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = np.random.normal(0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))
        self.wh2o = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))

        self.act_function = lambda x: scipy.special.expit(x)

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih1, inputs)
        hidden_ouputs1 = self.act_function(hidden_inputs1)

        hidden_inputs2 = np.dot(self.wh1h2, hidden_ouputs1)
        hidden_ouputs2 = self.act_function(hidden_inputs2)

        final_inputs = np.dot(self.wh2o, hidden_ouputs2)
        final_outputs = self.act_function(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih1, inputs)
        hidden_ouputs1 = self.act_function(hidden_inputs1)

        hidden_inputs2 = np.dot(self.wh1h2, hidden_ouputs1)
        hidden_ouputs2 = self.act_function(hidden_inputs2)

        final_inputs = np.dot(self.wh2o, hidden_ouputs2)
        final_outputs = self.act_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors2 = np.dot(self.wh2o.T, output_errors)
        hidden_errors1 = np.dot(self.wh1h2.T, hidden_errors2)

        self.wh2o += self.lr * np.dot(output_errors * final_outputs * (1-final_outputs), np.transpose(hidden_ouputs2))
        self.wh1h2 += self.lr * np.dot(hidden_errors2 * hidden_ouputs2 * (1-hidden_ouputs2), np.transpose(hidden_ouputs1))
        self.wih1 += self.lr * np.dot(hidden_errors1 * hidden_ouputs1 * (1-hidden_ouputs1), np.transpose(inputs))

        pass

    def save(self, name1, name2, name3):
        np.save(name1, self.wih1)
        np.save(name2, self.wh1h2)
        np.save(name3, self.wh2o)
        pass

    def load(self, name1, name2, name3):
        self.wih1 = np.load(name1 + '.npy')
        self.wh1h2 = np.load(name2 + '.npy')
        self.wh2o = np.load(name3 + '.npy')
        pass

    