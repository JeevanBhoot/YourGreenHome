import numpy as np
import scipy.special
import csv

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.act_function = lambda x: scipy.special.expit(x)

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.act_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.act_function(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.act_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.act_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(output_errors * final_outputs * (1-final_outputs), np.transpose(hidden_ouputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_ouputs * (1-hidden_ouputs), np.transpose(inputs))

        pass

    def save(self, name1, name2):
        np.save(name1, self.wih)
        np.save(name2, self.who)
        pass

    def load(self, name1, name2):
        self.wih = np.load(name1 + '.npy')
        self.who = np.load(name2 + '.npy')
        pass

    