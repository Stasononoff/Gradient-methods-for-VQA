from pennylane_cirq import ops as cirq_ops
import pennylane as qml

import numpy as np
from Circuit import *

config = {
    'dims': 4,
    'random_seed': 42,
    'eigenval' : 0,

    'circ': circuit_full_a,
    'dev': qml.device("cirq.mixedsimulator", wires=2),

    'nl_array' : np.linspace(0,3, num = 70), 
    'ly_array' : np.linspace(0,8, num = 70),

    # 'nl_array' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5], 
    # 'ly_array' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],

    'nl_array' : [0.01, 0.5], 
    'ly_array' : [0.001, 0.2],

    'opt_params' : {'stepsize': 0.02, 'beta1': 0.85, 'beta2': 0.9, 'eps': 1e-07},
    'adam' : True,
    'Stoh' : False,
    'QNG' : False,
    'init_params' : np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]),
    'steps' : 200
}
