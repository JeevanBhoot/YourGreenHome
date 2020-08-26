#Eight neural networks to estimate current and potential annual CO2 emissions, lighting cost, and heating + hot water cost. V6.
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

i_n = 14 #no. of input nodes
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

train_data_file = open('X:/Documents/Carbon Emissions Data/Data/TrainingData4Final.csv', 'r') #loading training data
train_data = train_data_file.readlines()
train_data_file.close()

def normalise_inputs(inputs):
    property_type = (inputs[0] - 2.5)/1.5 #normalising each piece of data into range [1,-1] approx
    built_form = (inputs[1] - 3.5)/2.5
    floor_area = (inputs[2] - 149.41)/142.69
    no_rooms = (inputs[3] - 15.5)/14.5
    no_fire = (inputs[4] - 4.95)/5
    hotwater =(inputs[5] - 5.95)/5
    windows = (inputs[6] - 3.97)/3
    walls = (inputs[7] - 1.99)
    roof = (inputs[8] - 2.5)/1.5
    heating = (inputs[9] - 12.5)/11.5
    storey_height = (inputs[10] - 4.374)/2.6
    photo_cells = inputs[11] * 0.5      
    ventilation = (inputs[12] - 1.99)
    year_band = (inputs[13] - 5.95)/5
    return [property_type, built_form, floor_area, no_rooms, no_fire, hotwater, windows, walls, roof, heating, storey_height, photo_cells, ventilation, year_band]

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
        for k in values:
            if k == ' ':
                k=0
        inputs = np.asfarray(values[:14])
        targets = np.asfarray(values)
        if targets[15] > targets[14] or (targets[15] != 1 and targets[15] == targets[14]) or targets[17] >= targets[16] or targets[19] >= targets[18] or targets[21] >= targets[20]:
            continue
        else:
            normal_inputs = normalise_inputs(inputs)
            n_current_nrg = (targets[14] - 0.07)/7 #normalising output/target values into range [0,0.99] approx
            n_potential_nrg = (targets[15] - 0.07)/7
            n_current_co2 = targets[16]/19
            n_potential_co2 = targets[17]/19        
            n_current_light = targets[18]/165
            n_potential_light = targets[19]/165
            n_current_hhw = targets[20]/1656
            n_potential_hhw = targets[21]/1656
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

NN1.save('a6', 'b6')
NN2.save('c6', 'd6')
NN3.save('e6', 'f6')
NN4.save('g6', 'h6')
NN5.save('i6', 'j6')
NN6.save('k6', 'l6')
NN7.save('m6', 'n6')
NN8.save('o6', 'p6')

print('The neural networks have successfully been trained.')