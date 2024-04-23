import numpy as np
import matplotlib.pyplot as plt
from Run import *
from BenchmarkFunctions import *


n = 50  # Number of Population
Func_name = 'F1'  # Name of the test function, range from F1-F14
MaxIt = 500  # Maximum number of iterations
# Load details of the selected benchmark function
lb, ub, dim, fobj = BenchmarkFunctions(Func_name)

Best_fitness, BestPositions, Convergence_curve = RUN(n, MaxIt, lb, ub, dim, fobj)  # Assuming RUN function is implemented

# Draw objective space
plt.figure()
plt.plot(Convergence_curve, color='r', linewidth=4)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.axis('tight')
plt.legend(['RUN'])
plt.show()