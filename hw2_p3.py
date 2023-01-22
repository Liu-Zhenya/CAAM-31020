
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# part a
# The template looks like this:
def derivatives(f,x):
    f_x = 0
    g = 0
    h = 0
    return f_x,g,h

def newton(f,x_0,k):
    x = x_0
    n = 0
    while(n <= k):
        f_x,g,h = derivatives(f,x)
        x = x + np.linalg.solve(-h,g)
        n = n + 1
    
    return x

# part b

# define the Gradient and Hessian matrix
x, y = symbols('x y')
f = symbols('f', cls=Function)
f = (1/2)*(12+x**2+((1+y**2)/x**2)+(((x*y)**2 + 100)/(x*y)**4))

G = Matrix([f]).jacobian(Matrix(list(f.free_symbols)))
H = hessian(f, [x, y])



def derivatives(f,X):
    x_1 = X[0]
    x_2 = X[1]
    f_x = f.subs([(x,x_1), (y,x_2)])
    g = G.subs(x,x_1).subs(y,x_2)
    h = H.subs(x,x_1).subs(y,x_2)

    return f_x,np.array(g).astype(np.float64),np.array(h).astype(np.float64)

def hw2_p3(x_0,k):
    X = x_0
    n = 0
    y_list = np.zeros(k)
    while(n < k):
        f_x,g,h = derivatives(f,X)
        y_list[n] = f_x
        tmp = g[0][0]
        g[0][0] = g[0][1]
        g[0][1] = tmp
        X = X + np.linalg.lstsq((-1)*h,g.T)[0].flatten('F')
        n = n + 1
    return X,y_list

start = np.array([3,2])
X,y_list = hw2_p3(start,20)

x_list = np.arange(20)

plt.plot(x_list,y_list)
plt.xlabel('iterations')
plt.ylabel('f(x)')
plt.title('start point at (3,2)')


# part c
start = np.array([3,4])
X,y_list = hw2_p3(start,20)

x_list = np.arange(20)

plt.figure()
plt.plot(x_list,y_list)
plt.xlabel('iterations')
plt.ylabel('f(x)')
plt.title('start point at (3,4)')


# part d

def hw2_p3_problem_d(x_0,k):
    X = x_0
    n = 0
    y_list = np.zeros(k)
    norm_list = [None] * 20
    yardstick = 10**(-8)
    key_iteration = 0
    while(n < k):
        f_x,g,h = derivatives(f,X)
        y_list[n] = f_x
        norm_list[n] = X
        tmp = g[0][0]
        g[0][0] = g[0][1]
        g[0][1] = tmp
        X = X + np.linalg.lstsq((-1)*h,g.T)[0].flatten('F')
        if(np.linalg.norm(y_list[n] - y_list[n-1]) < yardstick):
            key_iteration = n
            break
        n = n + 1
    return X,norm_list,key_iteration

start = np.array([3,2])
X,norm_list,key = hw2_p3_problem_d(start,20)
converge_rate = (np.linalg.norm(norm_list[key] - norm_list[key-1])/(np.linalg.norm(norm_list[key] - np.linalg.norm(norm_list[key-2]))**2))
print(converge_rate)