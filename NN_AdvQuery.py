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

i_n = 14 #no. of input nodes
h_n = 21 #no. of hidden nodes 11 ~ sqrt(i_n * o_n)
o_n = 1 #np. of output nodes
lr = 0.15 #learning rate
epochs = 1 #no. of epochs 

NN1 = NeuralNetwork(i_n, h_n, o_n, lr) #setting up neural networks
NN2 = NeuralNetwork(i_n, h_n, o_n, lr)
NN3 = NeuralNetwork(i_n, h_n, o_n, lr)
NN4 = NeuralNetwork(i_n, h_n, o_n, lr)
NN5 = NeuralNetwork(i_n, h_n, o_n, lr)
NN6 = NeuralNetwork(i_n, h_n, o_n, lr)
NN7 = NeuralNetwork(i_n, h_n, o_n, lr)
NN8 = NeuralNetwork(i_n, h_n, o_n, lr)

NN1.load('a6', 'b6')
NN2.load('c6', 'd6')
NN3.load('e6', 'f6')
NN4.load('g6', 'h6')
NN5.load('i6', 'j6')
NN6.load('k6', 'l6')
NN7.load('m6', 'n6')
NN8.load('o6', 'p6')

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

#Query neural networks

property_type = 4
built_form = 5
floor_area = 170
no_rooms = 9
no_fire = 1
hotwater = 6
windows = 3
walls = 1
roof = 1
heating = 22
storey_height = 2.4
photo_cells = 1
vent = 1
year_band = 9


inputs = [property_type, built_form, floor_area, no_rooms, no_fire, hotwater, windows, walls, roof, heating, storey_height, photo_cells, vent, year_band]
n_inputs = normalise_inputs(inputs)

n_c_nrg = float(NN1.query(n_inputs))
n_p_nrg = float(NN2.query(n_inputs))
n_c_co2 = float(NN3.query(n_inputs))
n_p_co2 = float(NN4.query(n_inputs))
n_c_light = float(NN5.query(n_inputs))
n_p_light = float(NN6.query(n_inputs))
n_c_hhw = float(NN7.query(n_inputs))
n_p_hhw = float(NN8.query(n_inputs))     

outputs = return_outputs([n_c_nrg, n_p_nrg, n_c_co2, n_p_co2, n_c_light, n_p_light, n_c_hhw, n_p_hhw])
print(outputs)