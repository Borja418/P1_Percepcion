import numpy as np

B = np.float32([[0,0,0], [0,0,-5], [1.5,0,-5], [1.5,0,-3], [1,0,-3], [0,0,-3], [2,0,-3], [2,0,0]]).reshape(-1,3)
O = np.float32([[0,0,0], [0,0,-5], [2,0,-5], [2,0,0]]).reshape(-1,3)
R = np.float32([[0,0,0], [0,0,-5], [2,0,-5], [2,0,-3], [0,0,-3], [1,0,-3], [2,0,0]]).reshape(-1,3)
J = np.float32([[0.5,0,-5], [2,0,-5], [1.25,0,-5], [1.25,0,0], [0,0,0], [0,0,-2]]).reshape(-1,3)
A = np.float32([[0,0,0], [0.5,0,-2.5], [1,0,-5], [1.5,0,-2.5], [2,0,0]]).reshape(-1,3)