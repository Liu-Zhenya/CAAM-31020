import scipy
import numpy as np
import matplotlib.pyplot as plt

# define Hessian matrix for Rosenbrock
def Hessian(x):
    n = int(len(x))
    H = np.zeros((n,n))

    H[0,0] = 1200*(x[0]**2)-400*x[1]+2
    H[0,1] = -400*(x[0])

    for i in range(1,n-1):

        H[i,i-1] = -400*x[i-1]
        H[i,i] = 1200*(x[i]**2)-400*x[i+1]+202
        H[i,i+1] = -400*x[i]
    
    H[n-1,n-2] = -400*x[n-2]
    H[n-1,n-1] = 200

    return H


# define Gradient for Rosenbrock

def Gradient_Rosenbrock(x):
    n = int(len(x))
    G = np.zeros(n)
    G[0] = 400*(x[0]**3)-400*x[0]*x[1]+2*x[0]-2
    for i in range(1,n-1):
        G[i] = 200*x[i] - 200*(x[i-1]**2) + 400*(x[i]**3) - 400*x[i]*x[i+1] + 2*x[i] - 2

    G[n-1] = 200*(x[n-1] - (x[n-2]**2))

    return G

# define function value of Rosenbrock

def f_Rosenbrock(x):
    f = 0
    n = int(len(x))

    for i in range(n-1):
        f = f + 100*(x[i+1] - (x[0]**2))**2 + (x[i] - 1)**2
    

    return f


# combine above parameters

def Rosenbrock(x):
    return f_Rosenbrock(x),Gradient_Rosenbrock(x),Hessian(x)
def trust_region_cg(x_in,tol,nabla):
    n = 0
    X = x_in
    f_x,g,B = Rosenbrock(X)
    Tol = tol
    number_evaluate = 0

    z = np.zeros(len(x_in))
    p = np.zeros(len(x_in))
    r = g
    d = -r

        

    
    if(np.linalg.norm(r) < Tol):
        return p,1
            
    while(True):

        j = 0

        if(j > 20):
            p = -g
            break

        if(d.T @ B @ d <= 0):
            tao = (-2*(d @ z) + np.sqrt(4*(d @ z)**2 - 4*(z @ z - nabla**2)*(d @ d)))/(2*(d @ d))
            p = z + tao*d
            break
        
        alpha = (r.T @ r)/(d.T @ B @ d)

        
        z_new = z + alpha*d

        if(np.linalg.norm(z_new) >= nabla):
            tao = (-2*(d @ z) + np.sqrt(4*(d @ z)**2 - 4*(z @ z - nabla**2)*(d @ d)))/(2*(d @ d))
            p = z + tao*d
            break
        r_new = r + alpha*(B @ d)

        if(np.linalg.norm(r_new) < Tol):
            p = z_new
            break

        beta = (r_new @ r_new)/(r @ r)
        d = -r_new + beta*d
        j = j + 1
        r = r_new
        z = z_new

    
    
    

    return p,number_evaluate

# Define quadratic model m_k
def m_k(x,p):
    f_x,g,h = Rosenbrock(x)

    m = f_x + np.dot(g,p) + 0.5*(p.T @ h @ p)

    return m

def Trust_Region(x,niter,eps):
    k = 0
    number_evaluation = 0

    delta_max = 0.5
    delta = delta_max

    g = Gradient_Rosenbrock(x)
    number_evaluation = number_evaluation + 1
    x_old = x

    
    while(k < niter and np.linalg.norm(g) > eps):
        p,counts = trust_region_cg(x,eps,delta)
        
        
        number_evaluation = number_evaluation + 3 + counts
    
        x_old = x
        x_new = x + p

        f_x = f_Rosenbrock(x)


        f_x_new = f_Rosenbrock(x_new)
        number_evaluation = number_evaluation + 2

        rho = (f_x - f_x_new)/(m_k(x,np.zeros(len(p))) - m_k(x,p))
        number_evaluation = number_evaluation + 2

        if rho < 0.25:
            delta = delta/4
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = np.minimum((2*delta),delta_max)

        if rho > 0.25:
            x = x_new

        k = k + 1
    
    converge = 1
    if(np.linalg.norm(x_old - x) > 0.01):
        converge = 0

    return x,number_evaluation,converge



    def Perturb_Hessian(x):
    n = int(len(x))
    H = np.zeros((n,n))

    H[0,0] = 1200*(x[0]**2)-400*x[1]+2+(1/50)
    H[0,1] = -400*(x[0]) + (1/50)

    for i in range(1,n-1):

        H[i,i-1] = -400*x[i-1] + (1/50)
        H[i,i] = 1200*(x[i]**2)-400*x[i+1]+202 + (1/50)
        H[i,i+1] = -400*x[i] + (1/50)
    
    H[n-1,n-2] = -400*x[n-2] + (1/50)
    H[n-1,n-1] = 200 + (1/50)

    return H


# define Gradient for Rosenbrock

def Perturb_Gradient_Rosenbrock(x):
    n = int(len(x))
    G = np.zeros(n)
    G[0] = 400*(x[0]**3)-400*x[0]*x[1]+2*x[0]-2+(1/50)*x[0]
    for i in range(1,n-1):
        G[i] = 200*x[i] - 200*(x[i-1]**2) + 400*(x[i]**3) - 400*x[i]*x[i+1] + 2*x[i] - 2+(1/50)*x[i]

    G[n-1] = 200*(x[n-1] - (x[n-2]**2))+(1/50)*x[n-1]

    return G

# define function value of Rosenbrock

def Perturb_f_Rosenbrock(x):
    f = 0
    n = int(len(x))

    for i in range(n-1):
        f = f + 100*(x[i+1] - (x[i]**2))**2 + (x[i] - 1)**2+(1/100)*x[i]**2
    

    return f


# combine above parameters

def Perturb_Rosenbrock(x):
    return Perturb_f_Rosenbrock(x),Perturb_Gradient_Rosenbrock(x),Perturb_Hessian(x)

def Perturb_trust_region_cg(x_in,tol,nabla):
    n = 0
    X = x_in
    f_x,g,B = Perturb_Rosenbrock(X)
    Tol = tol
    number_evaluate = 0

    z = np.zeros(len(x_in))
    p = np.zeros(len(x_in))
    r = g
    d = -r

        

    
    if(np.linalg.norm(r) < Tol):
        return p,1
            
    while(True):

        j = 0

        if(j > 20):
            p = -g
            break

        if(d.T @ B @ d <= 0):
            tao = (-2*(d @ z) + np.sqrt(4*(d @ z)**2 - 4*(z @ z - nabla**2)*(d @ d)))/(2*(d @ d))
            p = z + tao*d
            break
        
        alpha = (r.T @ r)/(d.T @ B @ d)

        
        z_new = z + alpha*d

        if(np.linalg.norm(z_new) >= nabla):
            tao = (-2*(d @ z) + np.sqrt(4*(d @ z)**2 - 4*(z @ z - nabla**2)*(d @ d)))/(2*(d @ d))
            p = z + tao*d
            break
        r_new = r + alpha*(B @ d)

        if(np.linalg.norm(r_new) < Tol):
            p = z_new
            break

        beta = (r_new @ r_new)/(r @ r)
        d = -r_new + beta*d
        j = j + 1
        r = r_new
        z = z_new

    
    
    

    return p,number_evaluate

def Perturb_Trust_Region(x,niter,eps):
    k = 0
    number_evaluation = 0

    delta_max = 0.5
    delta = delta_max

    g = Perturb_Gradient_Rosenbrock(x)
    number_evaluation = number_evaluation + 1
    x_old = x

    
    while(k < niter and np.linalg.norm(g) > eps):
        p,counts = Perturb_trust_region_cg(x,eps,delta)
        
        
        number_evaluation = number_evaluation + 3 + counts
    
        x_old = x
        x_new = x + p

        f_x = Perturb_f_Rosenbrock(x)


        f_x_new = Perturb_f_Rosenbrock(x_new)
        number_evaluation = number_evaluation + 2

        rho = (f_x - f_x_new)/(m_k(x,np.zeros(len(p))) - m_k(x,p))
        number_evaluation = number_evaluation + 2

        if rho < 0.25:
            delta = delta/4
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = np.minimum((2*delta),delta_max)

        if rho > 0.25:
            x = x_new

        k = k + 1
    
    converge = 1
    if(np.linalg.norm(x_old - x) > 0.01):
        converge = 0

    return x,number_evaluation,converge


def Convergence_Trust_Region(x,niter,eps):
    k = 0
    number_evaluation = 0

    delta_max = 0.5
    delta = delta_max

    g = Gradient_Rosenbrock(x)
    g_old = g
    number_evaluation = number_evaluation + 1
    x_old = x

    
    while(k < niter and np.linalg.norm(g) > eps):
        p,counts = trust_region_cg(x,eps,delta)
        
        
        number_evaluation = number_evaluation + 3 + counts
    
        x_old = x
        g_old = Gradient_Rosenbrock(x_old)
        x_new = x + p
        g = Gradient_Rosenbrock(x_new)

        f_x = f_Rosenbrock(x)


        f_x_new = f_Rosenbrock(x_new)
        number_evaluation = number_evaluation + 2

        rho = (f_x - f_x_new)/(m_k(x,np.zeros(len(p))) - m_k(x,p))
        number_evaluation = number_evaluation + 2

        if rho < 0.25:
            delta = delta/4
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = np.minimum((2*delta),delta_max)

        if rho > 0.25:
            x = x_new

        k = k + 1
    
    rate = np.linalg.norm(g)/np.linalg.norm(g_old)

    return rate

def Convergence_Perturb_Trust_Region(x,niter,eps):
    k = 0
    number_evaluation = 0

    delta_max = 0.5
    delta = delta_max

    g = Perturb_Gradient_Rosenbrock(x)
    g_old = g
    number_evaluation = number_evaluation + 1
    x_old = x

    
    while(k < niter and np.linalg.norm(g) > eps):
        p,counts = Perturb_trust_region_cg(x,eps,delta)
        
        
        number_evaluation = number_evaluation + 3 + counts
    
        x_old = x
        g_old = Perturb_Gradient_Rosenbrock(x_old)
        x_new = x + p
        g = Perturb_Gradient_Rosenbrock(x_new)

        f_x = Perturb_f_Rosenbrock(x)


        f_x_new = Perturb_f_Rosenbrock(x_new)
        number_evaluation = number_evaluation + 2

        rho = (f_x - f_x_new)/(m_k(x,np.zeros(len(p))) - m_k(x,p))
        number_evaluation = number_evaluation + 2

        if rho < 0.25:
            delta = delta/4
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = np.minimum((2*delta),delta_max)

        if rho > 0.25:
            x = x_new

        k = k + 1

    rate = np.linalg.norm(g)/np.linalg.norm(g_old)

    return rate