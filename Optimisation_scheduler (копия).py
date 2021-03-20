import datetime
import pickle
import numpy as np
from numpy import pi
import time

from functions import *
from Hamiltonian import *
from Circuit import *
from Optimisation_process import *
from Config import *


class scedule_program():
  def __init__(self, config):

    self.dims = config['dims'], 
    self.random_seed = config['random_seed'],
    self.eigenval = config['eigenval'],
    self.nl_array = config['nl_array'], 
    self.ly_array = config['ly_array'],

    self.circ = config['circ'],
    self.dev = config['dev'],

    self.opt_params = config['opt_params'],
    self.adam = config['adam'],
    self.Stoh = config['Stoh'],
    self.QNG = config['QNG'],
    self.init_params = config['init_params'],
    self.steps = config['steps'],
    self.result = []
    self.start_time = datetime.datetime.now()
    



  def start(self):

    print('start')

    result = []
    
    start_time = time.time()
    
    for ly in self.ly_array[0]:
      for noise_level in self.nl_array[0]:

        
        # print(self.dims, self.random_seed, noise_level, ly, self.ly_array, self.adam, self.init_params, self.nl_array, self.steps)
        obs_matrix = build_Hamiltonian(dims = self.dims[0], random_seed = self.random_seed[0],
                          eigenval = self.eigenval[0], noise_level = noise_level, ly = ly) 
        cost_fn = build_cost(obs_matrix, self.circ[0], self.dev)
        

        hist, theta_hist = search_minimum(opt_params = self.opt_params[0],
        cost = cost_fn, adam = self.adam[0], Stoh = self.Stoh[0], QNG = self.QNG[0],
        init_params = self.init_params[0], steps = self.steps[0])
        result.append([ly, noise_level,hist,theta_hist])
        print(ly, noise_level, hist[-1])

        #сохраняем промежуточный результат:
        print("--- %s seconds ---" % (time.time() - start_time))
        with open('results/half_result_' + str(self.steps[0]) + '_' + str(self.start_time)+'_.txt', 'wb') as filehandle:
          pickle.dump(result, filehandle)
    
    self.result = result



    return result

  def save(self):
    time_ = datetime.datetime.now()

    with open('results/result_'+ str(self.steps[0]) + '_' + str(time_)+'_.txt', 'wb') as filehandle:
      pickle.dump(self.result, filehandle)

      return 'saved'



