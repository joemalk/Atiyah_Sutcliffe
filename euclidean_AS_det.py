# Import the necessary modules/libraries
import numpy as np
import sympy as sp
from numpy import poly1d
from scipy import linalg
import math
import random
import itertools
from sympy import symarray
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# compute j of a vector in C2
def quatj(u):
    if u.size!=2:
        print("Error, the vector is not in C2")
        return
    if linalg.norm(u)==0:
        print("Error, the vector in C2 is 0")
        return
    return np.array([-u[1].conjugate(),u[0].conjugate()])

# Compute the stereographic projection of a non-zero vector in R3
def stereo(x):
    if x.size!=3:
        print("Error, the vector is not in R3")
        return
    if linalg.norm(x)==0:
        print("Error, the vector in R3 is 0")
        return
    r=linalg.norm(x)
    if x[2]!=r:
        return np.array([1,complex(x[0],x[1])/(r-x[2])])
    elif x[2]==r:
        return np.array([0,-1])

# Convert a non-zero vector in C2 into a degree less than or equal to 1 polynomial of one variable t
def poly(u):
    if u.size!=2:
        print("Error, the vector is not in C2")
        return
    if linalg.norm(u)==0:
        print("Error, the vector in C2 is 0")
        return
    return poly1d(np.array([u[0],-u[1]]), variable="t")

# Compute the 2 by 2 determinant of two vectors in u and v in C2
def det(u,v):
    mytemp = np.zeros((2,2),dtype=np.complex)
    mytemp[0,:] = u[:]
    mytemp[1,:] = v[:]
    return linalg.det(mytemp)

# given an np array with the n points in R3 as rows, compute the associated determinant
def MyDet(x):
    n=x.shape[0]
    if x.shape[1]!=3:
        print("Error, the number of columns of x must be 3")
        return
    
    p=np.zeros((n,n,2),dtype=np.complex)
    for i in range(n-1):
        for j in range(i+1,n):
            p[i,j,:]=stereo(-x[i,:]+x[j,:])
            p[j,i,:]=quatj(p[i,j,:])

    M=np.zeros((n,n),dtype=np.complex)

    for i in range(n):
        mypoly = poly1d([complex(1,0)],variable="t")
        for j in range(n):
            if (j!=i):
                mypoly=mypoly*poly(p[i,j,:])
        mycoeffs = mypoly.coeffs[::-1]
        M[i,0:mycoeffs.size]=mycoeffs[:]

    P = 1.
    for i in range(n-1):
        for j in range(i+1,n):
            P*=det(p[i,j],p[j,i])

    return linalg.det(M)*P**(-1)

def A(conf,indices):
    n = conf.shape[0]
    i,j,k,l = indices
    vij = list(conf[j,:]-conf[i,:])
    rij = sqrt(vij[0]**2+vij[1]**2+vij[2]**2)
    vkl = list(conf[l,:]-conf[k,:])
    rkl = sqrt(vkl[0]**2+vkl[1]**2+vkl[2]**2)
    return (vij[0]*vkl[0]+vij[1]*vkl[1]+vij[2]*vkl[2])*(rij*rkl)**(-1)

def dot_prod_sym(conf,*args):
    n = conf.shape[0]
    myPerms = list(itertools.permutations(range(n)))
    sum = 0
    for perm in myPerms:
        prod = 1
        for indices in args:
            i,j,k,l = indices
            prod *= A(conf, [perm[i],perm[j],perm[k],perm[l]])
        sum += prod
    return sum*factorial(n)**(-1)

# returns a pseudo-random number between -a and a
def rand(a):
    return 2*a*random.random()-a

# generate N configurations of n points having each coordinate
# between -a and +a, put these N configurations in an x with 3
# indices, such that the distances between 0 and x_i are all greater or equal to a*eps
# and the distances between x_i and x_j, and those between x_i and -x_j are all
# greater or equal to a*eps. compute the corresponding N determinants and put them in
# an np array called MyD
# return x and MyD.
# eps should be small enough so that there are configurations with those requirements, but not
# too small, otherwise one would get rounding errors
# as a rule of thumb, take a to be approximately between 1.5*n and 2*n, eps to be 0.1 for N <= 30
def test(n,N,a, eps):
    x=np.zeros((N,n,3),dtype=np.float)
    
    for k in range(N):
        for i in range(n):
            condition = True
            while condition:
                for j in range(3):
                    x[k,i,j]=rand(a)
                condition = (linalg.norm(x[k,i,:])<a*eps)
                for l in range(i):
                    condition = condition | (linalg.norm(x[k,i,:]-x[k,l,:])<a*eps) | (linalg.norm(x[k,i,:]+x[k,l,:])<a*eps)
    MyD = np.zeros([N], np.complex)
    for k in range(N):
        MyD[k] = MyDet(x[k,:,:])
    
    return [x,MyD]

# do a similar testing but for special configurations near
# a collinear one
def test2(n,N,a):
    x=np.zeros((N,n,3),dtype=np.float)
    for k in range(N):
        for i in range(n):
            x[k,i,:] = np.array([i+1, rand(a),0])
    
    MyD = np.zeros([N], np.complex)
    for k in range(N):
        MyD[k] = MyDet(x[k,:,:])
    
    return [x,MyD]

# find minimum of the real parts of elements of MyD and its position
def find_min_real(MyD):
    MyD_real = MyD[:].real
    pos = MyD_real.argmin()
    min_real = MyD_real[pos]
    return [min_real, pos]

# find maximum of the real parts of elements of MyD and its position
def find_max_real(MyD):
    MyD_real = MyD[:].real
    pos = MyD_real.argmax()
    min_real = MyD_real[pos]
    return [min_real, pos]

# find maximum of the absolute value of the imaginary parts of elements of MyD and its position
def find_max_abs_imag(MyD):
    MyD_imag = abs(MyD[:].imag)
    pos = MyD_imag.argmax()
    max_imag = MyD_imag[pos]
    return [max_imag, pos]