from dynamic_programming import *
import numpy as np
from tqdm import tqdm

# q1
# value1, policy1 = value_iteration(np.zeros((gridSize, gridSize)), reward_R=0, verbose=False)
# print(value1)
# print(policy1)

# q2
# value2, policy2 = value_iteration(np.zeros((gridSize, gridSize)), reward_R=-0.25, verbose=False)
# print(value2)
# print(policy2)


# q3
# for R in tqdm(np.arange(0, 1.5, 0.001)):
#     value3, policy3 = value_iteration(np.zeros((gridSize, gridSize)), reward_R=round(R, 4), verbose=False)
#     if (policy3[7][:5] == 'R').all():
#         print('Smallest Value of R:', R)
#         print(policy3)
#         break

# q4
value4, policy4 = value_iteration(np.zeros((gridSize, gridSize)), reward_R=4, verbose=False)
print(value4)
print(policy4)