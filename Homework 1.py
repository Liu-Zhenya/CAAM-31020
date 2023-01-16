import numpy as np
import timeit
import matplotlib.pyplot as plt


# Problem (a)
def initial(N):
    """
    Give a positive Integer N, return a random nxn matrix A, and nx1 vector b
    """
    A = np.random.rand(N,N)
    b = np.random.rand(N,1)
    return A,b

# Problem (b)
def time(N):
    A,b = initial(N)
    start_time =timeit.default_timer()
    np.linalg.solve(A,b)
    stop_time =timeit.default_timer()
    return stop_time - start_time

# Problem (c)
x_axis = np.zeros(10)
y_axis = np.zeros(10)

for i in range(10):
    x_axis[i] = np.floor(10000*(100**(-(9-i)/9)))
    
total_time = 0

for i in range(10):
    for j in range(5):
        total_time = total_time + time(int(x_axis[i]))
    y_axis[i] = total_time/5
    total_time = 0

print(f"a(n) is shown below: \n {y_axis}")

plt.loglog(x_axis,y_axis)
plt.xlabel("n")
plt.ylabel("a(n)")


#Problem d
def time_eigenvalue(N):
    A,b = initial(N)
    start_time =timeit.default_timer()
    np.linalg.eig(A)
    stop_time =timeit.default_timer()
    return stop_time - start_time

x_axis_eig = np.zeros(10)
y_axis_eig = np.zeros(10)

for i in range(10):
    x_axis_eig[i] = np.floor(1000*(100**(-(9-i)/9)))
    
total_time = 0

for i in range(10):
    for j in range(5):
        total_time = total_time + time_eigenvalue(int(x_axis_eig[i]))
    y_axis_eig[i] = total_time/5
    total_time = 0

print(f"b(n) is shown below: \n {y_axis_eig}")
plt.figure()
plt.loglog(x_axis_eig,y_axis_eig)
plt.xlabel("n")
plt.ylabel("b(n) for eigenvalue")

# Problem e
plt.figure()
plt.loglog(x_axis,y_axis,label='a(n)')
plt.loglog(x_axis_eig,y_axis_eig,label='b(n)')
plt.legend()