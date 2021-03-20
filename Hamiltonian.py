import pennylane as qml
from pennylane import expval, var

import numpy as np
from numpy import pi

from functions import *

def build_Hamiltonian(dims = 4, random_seed = 42, eigenval = 0, noise_level = 0, ly = 0.1):

  n = dims

  np.random.seed(random_seed)

  Bell_vec = np.array([1,0,0,1])/np.sqrt(2)

  Noise = (np.random.rand(4) - 0.5)*noise_level

  Bell_vec += Noise
  Bell_vec = Bell_vec/np.sqrt((Bell_vec**2).sum())
  rand_vec_list  = np.random.rand(n,4)


  rand_vec_list[0] = Bell_vec

  rand_vec_list = orthogonalize(rand_vec_list)
  rand_vec_list = permute_elems(old_list = rand_vec_list,elems_positions =  [2,3,0,1])
  #rand_vec_list, np.dot(rand_vec_list[2], rand_vec_list[1]), norm(rand_vec_list[3])

  ly_list = np.array([[5,0,0,0],
                      [0,3,0,0],
                      [0,0,ly,0],
                      [0,0,0,0]])
  
  # print(rand_vec_list.T , ly_list , rand_vec_list)

  H = rand_vec_list.T @ ly_list @ rand_vec_list

  return H


def build_cost(obs_matrix, circ, dev):
  obs = qml.Hermitian(obs_matrix, wires=[0, 1])
  H = qml.Hamiltonian((1, ), (obs, ))

  cost_fn = qml.ExpvalCost(circ, H, dev)
  return cost_fn



