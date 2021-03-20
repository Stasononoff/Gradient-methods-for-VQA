import pennylane as qml
from numpy import pi

@qml.template
def HWP(thetta, wires):
  qml.RY(thetta, wires = wires)
  qml.RZ(pi/2, wires = wires)
  qml.RY(thetta, wires = wires).inv()

@qml.template
def QWP(thetta, wires):
  qml.RY(thetta, wires = wires)
  qml.RZ(pi/4, wires = wires)
  qml.RY(thetta, wires = wires).inv()



def circuit_full_a(params, wires = 1):

  QWP(params[0], wires = 0)
  HWP(params[1], wires = 0)

  qml.PauliX(wires = 1)

  qml.CNOT(wires=[0,1])

  QWP(params[2], wires = 0)
  HWP(params[3], wires = 0)
  QWP(params[4], wires = 1)
  HWP(params[5], wires = 1)
