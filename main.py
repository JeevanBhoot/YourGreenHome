# coding: utf-8
#YourGreenHome web application.
#Developed by Jeevan Singh Bhoot.
#16 pre-trained neural networks for each of the 8 different outputs for both the advanced and basic calculator.

from flask import Flask, render_template, request
#from NeuralNetwork.NeuralNetwork import NeuralNetwork
import numpy as np
import scipy.special
import csv

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.i_n = inputnodes
        self.h_n = hiddennodes
        self.o_n = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0, pow(self.h_n, -0.5), (self.h_n, self.i_n))
        self.who = np.random.normal(0, pow(self.o_n, -0.5), (self.o_n, self.h_n))
        self.act_func = lambda x: scipy.special.expit(x)

        pass
    def train(self, inputs_lst, targets_lst):
        inputs = np.array(inputs_lst, ndmin=2).T
        targets = np.array(targets_lst, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.act_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.act_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(output_errors * final_outputs * (1-final_outputs), np.transpose(hidden_ouputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_ouputs * (1-hidden_ouputs), np.transpose(inputs))

        pass
    def query(self, inputs_lst):
        inputs = np.array(inputs_lst, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.act_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.act_func(final_inputs)

        return final_outputs
    def save(self, name1, name2):
        np.save(name1, self.wih)
        np.save(name2, self.who)
        pass
    def load(self, name1, name2):
        self.wih = np.load(name1 + '.npy')
        self.who = np.load(name2 + '.npy')
        pass

i_na = 14 #no. of input nodes - advanced
i_nb = 11 #np. of input nodes - basic

h_n = 21 #no. of hidden nodes 
o_n = 1 #np. of output nodes
lr = 0.15 #learning rate

NN1 = NeuralNetwork(i_na, h_n, o_n, lr) #setting up advanced neural networks
NN2 = NeuralNetwork(i_na, h_n, o_n, lr)
NN3 = NeuralNetwork(i_na, h_n, o_n, lr)
NN4 = NeuralNetwork(i_na, h_n, o_n, lr)
NN5 = NeuralNetwork(i_na, h_n, o_n, lr)
NN6 = NeuralNetwork(i_na, h_n, o_n, lr)
NN7 = NeuralNetwork(i_na, h_n, o_n, lr)
NN8 = NeuralNetwork(i_na, h_n, o_n, lr)

NN1b = NeuralNetwork(i_nb, h_n, o_n, lr) #setting up basic neural networks
NN2b = NeuralNetwork(i_nb, h_n, o_n, lr)
NN3b = NeuralNetwork(i_nb, h_n, o_n, lr)
NN4b = NeuralNetwork(i_nb, h_n, o_n, lr)
NN5b = NeuralNetwork(i_nb, h_n, o_n, lr)
NN6b = NeuralNetwork(i_nb, h_n, o_n, lr)
NN7b = NeuralNetwork(i_nb, h_n, o_n, lr)
NN8b = NeuralNetwork(i_nb, h_n, o_n, lr)

NN1.load('weights/a6', 'weights/b6') #loading advanced neural networks with pre-trained weights
NN2.load('weights/c6', 'weights/d6')
NN3.load('weights/e6', 'weights/f6')
NN4.load('weights/g6', 'weights/h6')
NN5.load('weights/i6', 'weights/j6')
NN6.load('weights/k6', 'weights/l6')
NN7.load('weights/m6', 'weights/n6')
NN8.load('weights/o6', 'weights/p6')

NN1b.load('weights/a7', 'weights/b7') #loading basic neural networks with pre-trained weights
NN2b.load('weights/c7', 'weights/d7')
NN3b.load('weights/e7', 'weights/f7')
NN4b.load('weights/g7', 'weights/h7')
NN5b.load('weights/i7', 'weights/j7')
NN6b.load('weights/k7', 'weights/l7')
NN7b.load('weights/m7', 'weights/n7')
NN8b.load('weights/o7', 'weights/p7')

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

def normalise_inputs_b(inputs):
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

def fiverange(x):
    y = 0.05*x
    a = round(x-y, 2)
    b = round(x+y, 2)
    return(str(a) + ' - ' + str(b))

prop_dic = {'flat':1, 'bungalow':2, 'maisonette':3, 'house':4}
build_dic = {'enclosed mid-terrace':1, 'enclosed end-terrace':2, 'mid-terrace':3, 'end-terrace':4, 'semi-detached':5, 'detached':6}
water_dic = {'community scheme + solar':1, 'from main system + solar':2, 'gas boiler/circulator + solar':3, 'electric immersion + solar':4,
'community scheme':5, 'from main system':6, 'electric heat pump':7, 'gas boiler/circulator':8, 'from secondary system':9, 'electric immersion':10, 'solid fuel boiler':11}
windows_dic = {'full triple glazing':1, 'partial triple glazing':2, 'full double glazing':3, 'full secondary glazing':4, 'partial double glazing':5, 'partial secondary glazing':6, 'full single glazing':7}
insu_dic = {'full insulation':1, 'partial insulation':2, 'no insulation':3}
heat_dic = {'biomass/wood + underfloor':1, 'biomass/wood + radiators':2, 'biomass/wood + room heaters':3, 'multi-fuel + underfloor':4, 'multi-fuel + radiators':5, 'multi-fuel + room heaters':6,
'community scheme + underfloor':7, 'community scheme + radiators':8, 'ground source heat pump + underfloor':9, 'ground source heat pump + radiators':10, 'ground source heat pump + warm air':11,
'water source heat pump + underfloor':12, 'water source heat pump + radiators':13, 'water source heat pump + warm air':14, 'air source heat pump + underfloor':15, 'air source heat pump + radiators':16,
'air source heat pump + warm air':17, 'electric boiler + underfloor':18, 'electric boiler + radiators':19, 'electric boiler + warm air':20, 'gas boiler + underfloor':21, 'gas boiler + radiators':22,
'gas boiler + warm air':23, 'electric heaters':24}
solar_dic = {'no':1, 'yes':-1}
vent_dic = {'no: natural':1, 'yes: extract only':2, 'yes: supply + extract':3}
age_dic = {'after 2006':1, '2003 - 2006':2, '1996 - 2002':3, '1991 - 1995':4, '1983 - 1990':5, '1976 - 1982':6, '1967 - 1975':7, '1950 - 1966':8, '1930 - 1949':9, '1900 - 1929':10, 'before 1900':11}

nrg_dic = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G'}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/advanced_output', methods=['POST'])
def advanced_output():
    property_type = prop_dic[str(request.form['property_type'])]
    build_form = build_dic[str(request.form['build_form'])]
    area = float(request.form['floor_area'])
    hab_rooms = float(request.form['hab_rooms'])
    open_fire = float(request.form['open_fire'])
    hotwater = water_dic[str(request.form['hotwater'])]
    windows = windows_dic[str(request.form['windows'])]
    walls = insu_dic[str(request.form['walls'])]
    roof = insu_dic[str(request.form['roof'])]
    heating = heat_dic[str(request.form['heating'])]
    height = float(request.form['height'])
    solar = solar_dic[str(request.form['solar'])]
    vent = vent_dic[str(request.form['vent'])]
    age = age_dic[str(request.form['age'])]
    inputs = [property_type, build_form, area, hab_rooms, open_fire, hotwater, windows, walls, roof, heating, height, solar, vent, age]
    n_inputs = normalise_inputs(inputs)
    n_c_nrg = float(NN1.query(n_inputs))
    n_p_nrg = float(NN2.query(n_inputs))
    n_c_co2 = float(NN3.query(n_inputs))
    n_p_co2 = float(NN4.query(n_inputs))
    n_c_light = float(NN5.query(n_inputs))
    n_p_light = float(NN6.query(n_inputs))
    n_c_hhw = float(NN7.query(n_inputs))
    n_p_hhw = float(NN8.query(n_inputs))
    y = '£'    
    z = 'tonnes per year'  
    outputs = return_outputs([n_c_nrg, n_p_nrg, n_c_co2, n_p_co2, n_c_light, n_p_light, n_c_hhw, n_p_hhw])
    a = fiverange(outputs[2])
    b = fiverange(outputs[3])
    c = fiverange(outputs[4])
    d = fiverange(outputs[5])
    e = fiverange(outputs[6])
    f = fiverange(outputs[7])
    return render_template('output.html', ncnrg=nrg_dic[outputs[0]], npnrg=nrg_dic[outputs[1]], ncco2=a, npco2=b, 
    ncl=c, npl=d, nchhw=e, nphhw=f, y=y, z=z) 

@app.route('/basic_output', methods=['POST'])
def basic_output():
    property_type = prop_dic[str(request.form['property_type'])]
    build_form = build_dic[str(request.form['build_form'])]
    hab_rooms = float(request.form['hab_rooms'])
    open_fire = float(request.form['open_fire'])
    hotwater = water_dic[str(request.form['hotwater'])]
    windows = windows_dic[str(request.form['windows'])]
    walls = insu_dic[str(request.form['walls'])]
    roof = insu_dic[str(request.form['roof'])]
    heating = heat_dic[str(request.form['heating'])]
    solar = solar_dic[str(request.form['solar'])]
    vent = vent_dic[str(request.form['vent'])]
    inputs = [property_type, build_form, hab_rooms, open_fire, hotwater, windows, walls, roof, heating, solar, vent]
    n_inputs = normalise_inputs_b(inputs)
    n_c_nrg = float(NN1b.query(n_inputs))
    n_p_nrg = float(NN2b.query(n_inputs))
    n_c_co2 = float(NN3b.query(n_inputs))
    n_p_co2 = float(NN4b.query(n_inputs))
    n_c_light = float(NN5b.query(n_inputs))
    n_p_light = float(NN6b.query(n_inputs))
    n_c_hhw = float(NN7b.query(n_inputs))
    n_p_hhw = float(NN8b.query(n_inputs))
    outputs = return_outputs([n_c_nrg, n_p_nrg, n_c_co2, n_p_co2, n_c_light, n_p_light, n_c_hhw, n_p_hhw])
    a = fiverange(outputs[2])
    b = fiverange(outputs[3])
    c = fiverange(outputs[4])
    d = fiverange(outputs[5])
    e = fiverange(outputs[6])
    f = fiverange(outputs[7])
    y = '£'    
    z = 'tonnes per year'  
    return render_template('output.html', ncnrg=nrg_dic[outputs[0]], npnrg=nrg_dic[outputs[1]], ncco2=a, npco2=b, 
    ncl=c, npl=d, nchhw=e, nphhw=f, y=y, z=z) 

if __name__ == '__main__':
    app.run(debug=True)