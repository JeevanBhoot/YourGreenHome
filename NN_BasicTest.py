#Eight neural networks to estimate current and potential annual CO2 emissions, lighting cost, and heating + hot water cost. V5.
#Training data. m = 498601. Locations: Leicester, Three Rivers, Cardiff, Manchester, County Durham, Isle of Wight,
#City of London, Birmingham, Liverpool, Sheffield, Dorset, Plymouth, Monmouthshire, Cambridge, Bracknell Forest,
#Slough, Amber Valley, Broxbourne and Harlow.
#Clean data. Outliers removed.
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

NN1 = NeuralNetwork(i_n, h_n, o_n, lr) #setting up neural networks
NN2 = NeuralNetwork(i_n, h_n, o_n, lr)
NN3 = NeuralNetwork(i_n, h_n, o_n, lr)
NN4 = NeuralNetwork(i_n, h_n, o_n, lr)
NN5 = NeuralNetwork(i_n, h_n, o_n, lr)
NN6 = NeuralNetwork(i_n, h_n, o_n, lr)
NN7 = NeuralNetwork(i_n, h_n, o_n, lr)
NN8 = NeuralNetwork(i_n, h_n, o_n, lr)

NN1.load('a7', 'b7')
NN2.load('c7', 'd7')
NN3.load('e7', 'f7')
NN4.load('g7', 'h7')
NN5.load('i7', 'j7')
NN6.load('k7', 'l7')
NN7.load('m7', 'n7')
NN8.load('o7', 'p7')

test_data_file = open('X:/Documents/Carbon Emissions Data/Data/TestingDataBasicFinal.csv', 'r') #loading test data
test_data = test_data_file.readlines()
test_data_file.close()

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

#Test neural network

current_nrg_score = []
potential_nrg_score = []
current_co2_error = []
potential_co2_error = []
current_light_error = []
potential_light_error = []
current_hhw_error = []
potential_hhw_error = []

for k in test_data:
    test_values = k.split(',')
    test_inputs = np.asfarray(test_values[:11])
    test_targets = np.asfarray(test_values[11:])
    if test_targets[1] > test_targets[0] or (test_targets[1] != 1 and test_targets[1] == test_targets[0]) or test_targets[3] >= test_targets[2] or test_targets[5] >= test_targets[4] or test_targets[7] >= test_targets[6]:
        continue
    else:
        n_test_inputs = normalise_inputs(test_inputs)
        n_test_c_nrg = float(NN1.query(n_test_inputs))
        n_test_p_nrg = float(NN2.query(n_test_inputs))
        n_test_c_co2 = float(NN3.query(n_test_inputs))
        n_test_p_co2 = float(NN4.query(n_test_inputs))
        n_test_c_light = float(NN5.query(n_test_inputs))
        n_test_p_light = float(NN6.query(n_test_inputs))
        n_test_c_hhw = float(NN7.query(n_test_inputs))
        n_test_p_hhw = float(NN8.query(n_test_inputs))                   
        test_outputs = return_outputs([n_test_c_nrg, n_test_p_nrg, n_test_c_co2, n_test_p_co2, n_test_c_light, n_test_p_light, n_test_c_hhw, n_test_p_hhw])
        if test_outputs[0] == test_targets[0]:
            current_nrg_score.append(1)
        else:
            current_nrg_score.append(0)
        if test_outputs[1] == test_targets[1]:
            potential_nrg_score.append(1)
        else:
            potential_nrg_score.append(0)
        c_co2_error = 100*abs(test_outputs[2] - test_targets[2]) / test_targets[2]
        p_co2_error = 100*abs(test_outputs[3] - test_targets[3]) / test_targets[3]
        c_light_error = 100*abs(test_outputs[4] - test_targets[4]) / test_targets[4]
        p_light_error = 100*abs(test_outputs[5] - test_targets[5]) / test_targets[5]
        c_hhw_error = 100*abs(test_outputs[6] - test_targets[6]) / test_targets[6]
        p_hhw_error = 100*abs(test_outputs[7] - test_targets[7]) / test_targets[7]
        current_co2_error.append(c_co2_error)
        potential_co2_error.append(p_co2_error)
        current_light_error.append(c_light_error)
        potential_light_error.append(p_light_error)
        current_hhw_error.append(c_hhw_error)
        potential_hhw_error.append(p_hhw_error)
        pass
    pass

c_nrg_accuracy = round(100*sum(current_nrg_score) / len(current_nrg_score), 2)
p_nrg_accuracy = round(100*sum(potential_nrg_score) / len(potential_nrg_score), 2)
c_co2_avgerror = round(sum(current_co2_error) / len(current_co2_error), 2)
p_co2_avgerror = round(sum(potential_co2_error) / len(potential_co2_error), 2)
c_light_avgerror = round(sum(current_light_error) / len(current_light_error), 2)
p_light_avgerror = round(sum(potential_co2_error) / len(potential_light_error), 2)
c_hhw_avgerror = round(sum(current_hhw_error) / len(current_hhw_error), 2)
p_hhw_avgerror = round(sum(potential_hhw_error) / len(potential_hhw_error), 2)
print(c_nrg_accuracy, p_nrg_accuracy, c_co2_avgerror, p_co2_avgerror, c_light_avgerror, p_light_avgerror, c_hhw_avgerror, p_hhw_avgerror)