import numpy as np
from numpy import pi
import time
import pennylane as qml

def search_minimum(opt_params, cost, adam = True, Stoh = False, QNG = False, init_params = (-0.3, -0.2, -0.1,  0.1,  0.2,  0.3), steps = 500):
  hist = []
  theta_hist = []
  

  ##start_time = time.time()
  # opt = qml.AdamOptimizer(**opt_params)
  # theta = init_params
  # theta_hist.append(theta)

  # for _ in range(steps):
  #   theta = opt.step(cost, theta)
  #   hist.append(cost(theta))
  #   theta_hist = np.vstack([theta_hist, theta])


  if (adam == True):
    opt = qml.AdamOptimizer(**opt_params)
  else:
    opt = qml.QNGOptimizer(**opt_params)

  theta = init_params
  theta_hist.append(theta)

  for _ in range(steps):

      if ((Stoh == True) | (QNG == True)) & (adam == True):
        theta_grad = opt.compute_grad(cost, theta)

        if Stoh == True:
          ind = np.unique(np.random.randint(0, 5, size =  3, dtype = 'int'))
          theta_grad[0][0][ind] = 0

        if QNG == True:
          theta_grad = np.dot(cost.metric_tensor([theta]),theta_grad[0][0])

        theta = opt.apply_grad(theta_grad, theta)


      elif (QNG == True) | (adam == True):
        theta = opt.step(cost, theta)


      hist.append(cost(theta))
      theta_hist = np.vstack([theta_hist, theta])
      

  # print("--- %s seconds ---" % (time.time() - start_time))
  return hist, theta_hist


