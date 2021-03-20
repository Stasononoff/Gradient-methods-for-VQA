import numpy as np
from numpy import pi
import time
from pennylane_cirq import ops as cirq_ops

import pennylane as qml
from pennylane import expval, var
from pennylane import DepolarizingChannel,  PhaseDamping, GeneralizedAmplitudeDamping


import datetime
import pickle

from Optimisation_scheduler import *
from Config import *


       
## dev_H = qml.device("cirq.mixedsimulator", wires=2,  analytic=False)
# dev_H_a = qml.device("cirq.mixedsimulator", wires=2)
## dev_H_a = qml.device("default.qubit", wires=2)



#config['dev'] = dev_H_a
#config['circ'] = 



program = scedule_program(config)
result = program.start()
program.save()
