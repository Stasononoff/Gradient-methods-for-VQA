import numpy as np
from numpy import pi

##########
def norm(vec):
  return np.sqrt((vec*vec).sum())
############
def normalize(vec):
  return vec/norm(vec)
###############
def orthogonalize(vec_list):

  vec_list[0] = normalize(vec_list[0])
  for ind in range(1,len(vec_list)):
    for ind2 in range(ind):
      vec_list[ind] = vec_list[ind] - np.dot(vec_list[ind], vec_list[ind2])/(norm(vec_list[ind2]))**2 * vec_list[ind2]
    vec_list[ind] = normalize(vec_list[ind])

  return vec_list

###############
def permute_elems(old_list, elems_positions):
  if len(old_list) == len(elems_positions):
    new_list = np.zeros((len(old_list),len(old_list)))
    for i in range(len(old_list)):
      new_list[i] = old_list[elems_positions[i]]
  else:
    print('Ошибка: Размеры списков не совпадают')

  return new_list

######################
def unique_comb(ly_list, ny_list):
    param_tuple_list = []
    for ly in ly_list:
        for ny in ny_list:
            param_tuple_list.append([ly,ny])
    return param_tuple_list
