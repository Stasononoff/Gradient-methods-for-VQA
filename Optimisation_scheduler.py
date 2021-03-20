import datetime
import pickle
import numpy as np
from numpy import pi
import time

from math import sqrt
from joblib import Parallel, delayed

from functions import *
from Hamiltonian import *
from Circuit import *
from Optimisation_process import *
from Config import *

    
def param_set_optim(param_set, dims, random_seed, eigenval, circ, dev, opt_params, adam, Stoh, QNG, init_params, steps, start_time):
  ly, noise_level = param_set[0], param_set[1]
  # print(self.dims, self.random_seed, noise_level, ly, self.ly_array, self.adam, self.init_params, self.nl_array, self.steps)
  obs_matrix = build_Hamiltonian(dims = dims, random_seed = random_seed,
                          eigenval = eigenval, noise_level = noise_level, ly = ly) 
  cost_fn = build_cost(obs_matrix, circ, dev)
        

  hist, theta_hist = search_minimum(opt_params = opt_params,
  cost = cost_fn, adam = adam, Stoh = Stoh, QNG = QNG,
  init_params = init_params, steps = steps)
  #result.append([ly, noise_level,hist,theta_hist])
  print(ly, noise_level, hist[-1])

  #сохраняем промежуточный результат:
  print("--- %s seconds ---" % (time.time() - start_time))
  #with open('results/half_result_' + str(self.steps[0]) + '_' + str(self.start_time)+'_.txt', 'wb') as filehandle:
  #  pickle.dump(result, filehandle)
  return [ly, noise_level, hist,theta_hist]



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
    
	
    param_list = unique_comb(self.ly_array[0], self.nl_array[0])  
    
	
    result = Parallel(n_jobs=42, verbose=10)(delayed(param_set_optim)(param_set, self.dims[0], self.random_seed[0],
    self.eigenval[0], self.circ[0], self.dev, self.opt_params[0], self.adam[0], self.Stoh[0],
    self.QNG[0], self.init_params[0], self.steps[0], start_time) for param_set in param_list)

        
      
    
    self.result = result



    return result

  def save(self):
    time_ = datetime.datetime.now()

    with open('results/result_'+ str(self.steps[0]) + '_' + str(time_)+'_.txt', 'wb') as filehandle:
      pickle.dump(self.result, filehandle)

      return 'saved'



