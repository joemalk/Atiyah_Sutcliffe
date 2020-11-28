#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import the necessary modules/libraries
import numpy as np
import sympy as sp
from numpy import poly1d
from scipy import linalg
from scipy.special import binom
import math
from random import random
import itertools
from sympy import symarray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import collections  as mc
import pickle

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

# Compute the null direction from xa to xb
def null_direction(xa, xb):
    if (xa.shape != (2,2)) or (xb.shape != (2,2)):
        print("Error: xa and xb must both be 2 by 2 arrays.")
        return
    cab = 0.5*(xa+xb)
    dab = 0.5*(xb-xa)
    dab_mod = (np.linalg.inv(cab)).dot(dab)
    factor = - (1./(2.*np.linalg.det(dab_mod)))
    la = factor*(np.trace(dab_mod) + np.sqrt((np.trace(dab_mod))**2-4.*                            np.linalg.det(dab_mod)))
    vab = cab + la*dab
    return (2./np.trace(vab))*vab

# Compute all null directions from xa to xb, for a different from b
def all_null_directions(x, space='euclidean'):
    n = x.shape[0]
    p=np.zeros((n,n,2),dtype=np.complex)
    
    if space=='euclidean':
        for i in range(n-1):
            for j in range(i+1,n):
                p[i,j,:]=stereo(-x[i,:]+x[j,:])
                p[j,i,:]=quatj(p[i,j,:])
        return p
    
    elif space=='hyperbolic':
        for a in range(n):
            for b in range(n):
                if a != b:
                    null_dir = null_direction(x[a],x[b])
                    first_col = null_dir[:,0]
                    second_col = null_dir[:,1]
                    norm_first_col = np.linalg.norm(first_col)
                    if norm_first_col > 1.e-8:
                        p[a,b,:] = first_col/norm_first_col
                    else:
                        norm_second_col = np.linalg.norm(second_col)
                        p[a,b,:] = second_col/norm_second_col
        return p
    
    else:
        print("Is the space euclidean or hyperbolic?")
    
# generate N configurations of n points in space (Euclidean/hyperbolic)
def generate_random_confs(n,N,space='euclidean'):
    confs = []
    if space == 'euclidean':
        return [6.*np.random.rand(n,3) - 3. for k in range(N)]
    elif space == 'hyperbolic':
        for l in range(N):
            conf = []
            for a in range(n):
                x = np.eye(2) + (random()-0.5)*np.array([[0.,1.],[1.,0.]]) +                    (random()-0.5)*np.array([[0.,-1.j],[1.j,0.]]) +                    (random()-0.5)*np.array([[1.,0.],[0.,-1.]])
                conf.append(x)
            confs.append(conf)
        return np.array(confs)
    else:
        print("Is the space euclidean or hyperbolic?")
        
# given an np array with the n points in R3 as rows, compute the associated determinant
def AS_det(x, space='euclidean'):
    n=x.shape[0]
    
    p = all_null_directions(x, space)
    
#     p=np.zeros((n,n,2),dtype=np.complex)
#     for i in range(n-1):
#         for j in range(i+1,n):
#             p[i,j,:]=stereo(-x[i,:]+x[j,:])
#             p[j,i,:]=quatj(p[i,j,:])

    M=np.zeros((n,n),dtype=np.complex)

    if (space == 'euclidean') and (x.shape[1]!=3):
        print("Error, the number of columns of x must be 3")
        return
    
    if (space == 'hyperbolic') and ((x.shape[1], x.shape[2]) != (2,2)):
        print("Error, each entry of x must be an hermitian 2 by 2 matrix")
        return
    
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


# returns a pseudo-random number between -a and a
def rand(a):
    return 2*a*random.random()-a

# convert a subset of the set of pairs of indices to a graph
def graph_from_subset(subset,n):
    Ga = {}
    for a in range(n):
        for b in range(a+1,n):
            if (a,b) in subset:
                Ga[a] = Ga.get(a,set()).union(set({b}))
                Ga[b] = Ga.get(b,set()).union(set({a}))
    return Ga

# draw a graphical representation of a graph, represented by a subset
# of the set of all pairs of indices
def draw_graph(subset,n):
    points = np.zeros((n,2), dtype=np.float)
    for i in range(n):
        points[i,:] = [np.cos(2.*np.pi*i/float(n)),np.sin(2.*np.pi*i/float(n))]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter(points[:,0], points[:,1])
    offset = np.array([0.,0.05])
    ax.annotate('0', points[0]+offset)
    for i in range(n):
        ax.annotate(str(i), points[i]+offset)
    for pair in subset:
        mypoints = np.concatenate([points[pair[0]].reshape(1,-1),points[pair[1],:].reshape(1,-1)],axis=0)
        plt.plot(mypoints[:,0], mypoints[:,1])
        
# the trace function used in graph evaluation maps
def trace(outer_prod, i,j):
    n_indices = len(outer_prod.shape)
    if (not (0 <= i < n_indices)) or (not (0 <= j < n_indices)):
        print("Error: indices out of range when taking trace.")
        return
    inds1 = []
    for k in range(n_indices):
        if k == i:
            inds1.append(0)
        elif k == j:
            inds1.append(1)
        else:
            inds1.append(slice(2))
    inds1 = tuple(inds1)
    inds2 = []
    for k in range(n_indices):
        if k == i:
            inds2.append(1)
        elif k == j:
            inds2.append(0)
        else:
            inds2.append(slice(2))
    inds2 = tuple(inds2)
    return outer_prod[inds1]-outer_prod[inds2]

def compute_ind_i(a, vertices, valences):
    ii = 0
    index_vert = vertices.index(a)
    for k in range(index_vert):
        ii += valences[k]
    return ii  

# return the graph evaluation of the graph, given by a subset, at a configuration
# of n distinct points in space (Euclidean/hyperbolic)
def graph_eval(x, subset, space='euclidean'):
    if subset == []:
        return 1.
    n = x.shape[0]
    p = all_null_directions(x, space)
    Ga = graph_from_subset(subset,n)
    vertices = list(Ga.keys())
    vertices.sort()
    valences = [len(Ga.get(v,[])) for v in vertices]
    n_vertices = len(vertices)
    sym_tensors = []
    for k in range(n_vertices):
        valence = valences[k]
        sym_tens = np.zeros(tuple((2) for k in range(valence)),dtype=np.complex)
        for perm in itertools.permutations([p[vertices[k],nbr] for nbr in Ga[vertices[k]]]):
            tmp_tens = perm[0]       
            for i in range(1,len(perm)):
                tmp_tens = np.multiply.outer(tmp_tens,perm[i])
            sym_tens += tmp_tens 
#         sym_tens /= math.factorial(valence)
        sym_tensors.append(sym_tens)
    outer_prod = sym_tensors[0]
    for i in range(1,len(sym_tensors)):
        outer_prod = np.multiply.outer(outer_prod,sym_tensors[i])
    for pair in subset:
        a,b = pair
        i,j = compute_ind_i(a,vertices,valences),compute_ind_i(b,vertices,valences)
        valences[vertices.index(a)]-=1
        valences[vertices.index(b)]-=1
        outer_prod = trace(outer_prod,i,j)
    den = 1.
    for pair in subset:
        a,b = pair
        den *= det(p[a,b],p[b,a])
    return outer_prod/den

# evaluate the minimal graph evaluation of the given graph at
# a collinear configuration
def eval_col(n, subset, space='euclidean'):
    if space == 'euclidean':
        x_col = [[float(k),0.,0.] for k in range(n)]
    elif space == 'hyperbolic':
        x_col = np.array([np.eye(2,dtype=np.complex) + float(k)/float(n)*np.array([[1.,0.],[0.,-1.]]) for k in range(n)])
    else:
        print("The space must be either euclidean or hyperbolic")
        return
    x_col_permuted = list(itertools.permutations(x_col))
    eval_cols = []
    for x_col in x_col_permuted:
        x_col = np.array(x_col)
        eval_cols.append(graph_eval(x_col, subset, space))
    eval_cols = np.array(eval_cols)
    return min(np.real(eval_cols))


def test(n,N,subset, space='euclidean'):
    confs = generate_random_confs(n,100,space)
    graph_evals = []
    for x in confs:
        graph_evals.append(graph_eval(x,subset))
    graph_evals = np.array(graph_evals)
    return graph_evals

# testing that my functions behave the way they should
def tests():
    n = 4
    inds = list(range(n))
    pairs_of_inds = list(itertools.combinations(inds,2))
    n_choose_2 = len(pairs_of_inds)
    full_subset = pairs_of_inds
    
    space1 = 'euclidean'
    euc_confs = generate_random_confs(n,100,space1)
    error1 = max(abs(np.array([AS_det(x,space1) - graph_eval(x,full_subset,space1)/144. for x in euc_confs])))
    if error1 < 1.e-9:
        print("Test 1 (euclidean) OK.")
    else:
        print("Test 1 (euclidean) failed.")
    
    space2 = 'hyperbolic'
    hyp_confs = generate_random_confs(n,100,space2)
    error2 = max(abs(np.array([AS_det(x,space2) - graph_eval(x,full_subset,space2)/144. for x in hyp_confs])))
    if error2 < 1.e-9:
        print("Test 2 (hyperbolic) OK.")
    else:
        print("Test 2 (hyperbolic) failed.")


# In[3]:

# generate 20 random graphs, and for each one of these graphs, generate
# a sample set of 1000 configurations of n distinct points in space
# (Euclidean/hyperbolic), calculate their graph evaluations, and then
# compute the minimal value of the real parts of these graph evaluations
# for each one of these sample sets
def lower_bound_graph_eval(n, space):
    inds = list(range(n))
    pairs_of_inds = list(itertools.combinations(inds,2))
    n_choose_2 = len(pairs_of_inds)
    full_subset = pairs_of_inds
    subsets = []
    results = []
    for count in range(20):
        subset = list(itertools.compress(pairs_of_inds,np.array(np.random.randint(0,2,n_choose_2),np.bool)))
        # subset2 = subset1[0:-1]
        subsets.append(subset)
        draw_graph(subset,n)
        confs = generate_random_confs(n,1000,space)
        graph_evals = np.array([graph_eval(x,subset,space) for x in confs])
        eval_collinear = eval_col(n,subset,space)
        results.append([min(np.real(graph_evals)), eval_collinear])
    results = np.array(results)
    return min(results[:,0]), results


# In[6]:


lower_bound_graph_eval(3,'euclidean')


# In[4]:


lower_bound_graph_eval(4,'euclidean')


# In[7]:


lower_bound_graph_eval(5,'euclidean')


# In[6]:


lower_bound_graph_eval(6,'euclidean')


# In[7]:


lower_bound_graph_eval(3,'hyperbolic')


# In[8]:


lower_bound_graph_eval(4,'hyperbolic')


# In[9]:


lower_bound_graph_eval(5,'hyperbolic')


# In[10]:


lower_bound_graph_eval(6,'hyperbolic')


# In[ ]:




