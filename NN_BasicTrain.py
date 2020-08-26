#Eight neural networks to estimate current and potential annual CO2 emissions, lighting cost, and heating + hot water cost. V6 Basic.
#Training data. m = 498601. Locations: Leicester, Three Rivers, Cardiff, Manchester, County Durham, Isle of Wight,
#City of London, Birmingham, Liverpool, Sheffield, Dorset, Plymouth, Monmouthshire, Cambridge, Bracknell Forest,
#Slough, Amber Valley, Broxbourne and Harlow.
#Clean data. Further outliers removed.
#Testing data. m = 66749. Locations: Nottingham and Reading.

#Set up neural networks

from NeuralNetwork.NeuralNetwork import NeuralNetwork
import numpy as np
import scipy.special
import csv

i_n = 11 #no. of input nodes
h_n = 21 #no. of hidden nodes 11 ~ sqrt(i_n * o_n)
o_n = 1 #np. of output nodes
lr = 0.15 #learning rate
epochs = 5 #no. of epochs 

NN1 = NeuralNetwork(i_n, h_n, o_n, lr) #setting up neural networks
NN2 = NeuralNetwork(i_n, h_n, o_n, lr)
NN3 = NeuralNetwork(i_n, h_n, o_n, lr)
NN4 = NeuralNetwork(i_n, h_n, o_n, lr)
NN5 = NeuralNetwork(i_n, h_n, o_n, lr)
NN6 = NeuralNetwork(i_n, h_n, o_n, lr)
NN7 = NeuralNetwork(i_n, h_n, o_n, lr)
NN8 = NeuralNetwork(i_n, h_n, o_n, lr)

train_data_file = open('X:/Documents/Carbon Emissions Data/Data/TrainingDataBasicFinal.csv', 'r') #loading training data
train_data = train_data_file.readlines()
train_data_file.close()

def normalise_inputs(inputs):
    property_type = (inputs[0] - 2.5)/1.5 #normalising each piece of data into range [1,-1] approx
    built_form = (inputs[1] - 3.5)/2.5
    no_rooms = (inputs[2] - 15.5)/14.5
    no_fire = (inputs[3] - 4.95)/5
    hotwater =(inputs[4] - 5.95)/5
    windows = (inputs[5] - 3.97)/3
    walls = (inputs[6] - 1.99)
    roof = (inputs[7] - 2.5)/1.5
    heating = (inputs[8] - 12.5)/11.5
    photo_cells = inputs[9] * 0.5      
    ventilation = (inputs[10] - 1.99)
    return [property_type, built_form, no_rooms, no_fire, hotwater, windows, walls, roof, heating, photo_cells, ventilation]

def return_outputs(n_outputs):
    current_nrg = round(float(7*n_outputs[0] + 0.07))
    potential_nrg = round(float(7*n_outputs[1] + 0.07))
    current_co2 = float(19*n_outputs[2])
    potential_co2 = float(19*n_outputs[3])
    current_lighting = float(165*n_outputs[4])
    potential_lighting = float(165*n_outputs[5])
    current_hhw = float(1656*n_outputs[6])
    potential_hhw = float(1656*n_outputs[7])
    return [current_nrg, potential_nrg, current_co2, potential_co2, current_lighting, potential_lighting, current_hhw, potential_hhw]

#Train neural networks

for i in range(epochs):
    for j in train_data:
        values = j.split(',')
        inputs = np.asfarray(values[:11])
        targets = np.asfarray(values)
        if targets[12] > targets[11] or (targets[12] != 1 and targets[12] == targets[11]) or targets[14] >= targets[13] or targets[16] >= targets[15] or targets[18] >= targets[17]:
            continue
        else:
            normal_inputs = normalise_inputs(inputs)
            n_current_nrg = (targets[11] - 0.07)/7 #normalising output/target values into range [0,0.99] approx
            n_potential_nrg = (targets[12] - 0.07)/7
            n_current_co2 = targets[13]/19
            n_potential_co2 = targets[14]/19        
            n_current_light = targets[15]/165
            n_potential_light = targets[16]/165
            n_current_hhw = targets[17]/1656
            n_potential_hhw = targets[18]/1656
            NN1.train(normal_inputs, n_current_nrg)
            NN2.train(normal_inputs, n_potential_nrg)
            NN3.train(normal_inputs, n_current_co2)
            NN4.train(normal_inputs, n_potential_co2)
            NN5.train(normal_inputs, n_current_light)
            NN6.train(normal_inputs, n_potential_light)
            NN7.train(normal_inputs, n_current_hhw)
            NN8.train(normal_inputs, n_potential_hhw)
            pass
        pass
    pass                                 

NN1.save('a7', 'b7')
NN2.save('c7', 'd7')
NN3.save('e7', 'f7')
NN4.save('g7', 'h7')
NN5.save('i7', 'j7')
NN6.save('k7', 'l7')
NN7.save('m7', 'n7')
NN8.save('o7', 'p7')

print('The neural networks have successfully been trained.')