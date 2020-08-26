NEURAL NETWORKS
Sixteen neural networks - eight for each of the advanced and basic calculator. Currently trained on data from 778 000 homes across England and Wales, and tested on data
from 67 000 different homes. Uses self-made neural network class, which has four functions: query, train, save and load.

TRAINING
Train advanced calculator and save weights using NN_AdvTrain.py. Change path of training data on line 30. Rename path/names of weights on lines 96 - 103, as desired.
Trainbasic calculator and save weights using NN_BasicTrain.py. Change path of training data on line 30. Rename path/names of weights on lines 90 - 97, as desired.

TESTING
Test calculators with NN_AdvTest.py and NN_BasicTest.py. Load weights of neural networks with previously set path and filename. Edit filepath of testing data on line 38 of both calculators.

WEBAPP
Webapp runs through app.py. Running app.py will start a development server on local host, which can be accessed with http://127.0.0.1:5000/ .

REQUIREMENTS
Enter pip install -r requirements.txt in console.
