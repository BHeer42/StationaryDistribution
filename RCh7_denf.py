# -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 11:16:54 2020


author: Burkhard Heer

book: Dynamic General Equilibrium Modeling: Computational
        Methods and Applications, with Alfred Maussner

Chapter 7.2., distribution function

Algorithm 7.2.3 in Heer/Maussner (2009)

computation of the policy functions and the distribution function
in the heterogenous-agent neoclassical growth model
with value function iteration

Linear interpolation between grid points

Maximization of value function: golden section search method


"""

# Part 1: import libraries
import numpy as np
from scipy.linalg import inv
from scipy import interpolate
import time
import matplotlib.pyplot as plt
import math

log = math.log

# Part 2: functions
# equivec1 computes the ergodic distribution
# see pages 654-656 in Heer/Maussner (2009)
# on DSGE modeling, 2nd edition
def equivec1(p):
    nrows,ncols = p.shape
    for i in range(nrows):
        p[i,i] = p[i,i]-1
    
    q = p[:,0:nrows-1]
    # appends column vector
    q = np.c_[q,np.ones(nrows)]  
    x = np.zeros(nrows-1)
    # appends element in row vector
    x = np.r_[x,1]
    y =  np.transpose(x) @ inv(q)
    return np.transpose(y)


#
# wage function
def wage_rate(k,l):
    return (1-alpha) * k**alpha * l**(-alpha)

# interest rate function
def interest_rate(k,l):
    return alpha * k**(alpha - 1) * l**(1-alpha) - delta


# utility function 
def utility(x): 
    if sigma==1:
        return  log(x) 
    else:
        return x**(1-sigma) / (1-sigma)


# right-hand-side of the Bellman equation
# input:
# a0 - wealth a in period t
# a1 - wealth a' in period t+1
# y - employment status in period t
# output:
# rhs of Bellman eq.    
def bellman(a0,a1,y):
   if y==0:     # y=0: employed, y=1: unemployed
      c = (1+(1-tau)*r) * a0 + (1-tau)*w - a1
   else:
       c = (1+(1-tau)*r)*a0 + b - a1
   
   if c<0:
      return negvalue 
   
   if a1>amax:
      return a1**2*negvalue
  
   # output: right-hand side of the Bellman equation  
   return utility(c) + beta*(prob[y,0]*value(a1,0) + prob[y,1]*value(a1,1)) 



# value interpolates value function vold[a,e] at a
def value(x,y):
    if y==0:
        return ve_polate(x)
    else:
        return vu_polate(x)
    

# value1 rewrites the Bellman function
# as a function of one variable x=a', the next-period asset level
# for given present period assets a[i] and employment status e
# -> function in one argument x necessary for applying
# Golden Section Search
def value1(x):
    return bellman(ainit,x,e)

# searches the MAXIMUM using golden section search
# see also Chapter 11.6.1 in Heer/Maussner, 2009,
# Dynamic General Equilibrium Modeling: Computational
# Methods and Applications, 2nd ed. (or later)
def GoldenSectionMax(f,ay,by,cy,tol):
    r1 = 0.61803399 
    r2 = 1-r1
    x0 = ay
    x3 = cy  
    if abs(cy-by) <= abs(by-ay):
        x1 = by 
        x2 = by + r2 * (cy-by)
    else:
        x2 = by 
        x1 = by - r2 * (by-ay)
    
    f1 = - f(x1)
    f2 = - f(x2)

    while abs(x3-x0) > tol*(abs(x1)+abs(x2)):
        if f2<f1:
            x0 = x1
            x1 = x2
            x2 = r1*x1+r2*x3
            f1 = f2
            f2 = -f(x2)
        else:
            x3 = x2
            x2 = x1
            x1 = r1*x2+r2*x0
            f2 = f1
            f1 = -f(x1)
            
    if f1 <= f2:
        xmin = x1
    else:
        xmin = x2
    
    return xmin

def testfunc(x):
    return -x**2 + 4*x + 6

# test goldensectionsearch
xmax = GoldenSectionMax(testfunc,-2.2,0.0,10.0,0.001)
print(xmax)

start_time = time.time()

# Part 3: Numerical parameters
tol = 0.001             # stopping criterion for final solution of K 
tol1 = 1e-7             # stopping criterion for golden section search 
tolg = 0.0000001        # stopping criterion for distribution function 
negvalue = -1e10             # initialization for the value function 
eps = 0.05              # minimum distance between grid point amin and amin+varepsilon
                         # in order to check if amin is binding lower bound
nit = 100                # maximum number of maximum iterations over value function 
ngk = 0                  # initial number of iterations over distribution function 
crit1 = 1 + tol          # percentage deviation in the iteration over the aggregate capital stock
critg = 1 + tol          # deviation of two successive solutions for distribution function:
                         # kritg=sumc(abs(gk0-gk));
nq = 70                  # maximum number of iterations over K, outer loop

psi1 = 0.95              # linear update of K and tau in the outer loop
                         # K^{i+1} = psi1 * K^i + (1-psi1) * K^{i*}, where K^{i*} new solution in iteration i
kbarq = np.zeros(nq)
k1barq = np.zeros(nq)           # convergence of kk1 ?
kritvq = np.zeros(nq)       # convergence of value function 
crit = 1+tol                    # error in value function: crit=meanc(abs(vold-v));


# Part 4: Parameterization of the model
alpha = 0.36
beta = 0.995
delta = 0.005
sigma = 2
tau = 0.02
r = 0.005
rep = 0.25
pp0 = np.array([[0.9565,0.0435],[0.5,0.5]]) # for the computaton of the
                                            # ergodic distribution
prob = pp0 + 0  # entries are the same as in pp0



# asset grid: Value function
amin = -2               # lower bound asset grid over value function
amax = 3000             # upper bound    
na = 201                 # number of equispaced points
a =  np.linspace(amin, amax, na)   # asset grid 


# asset grid: Distribution function: Step 5.1 from Algorithm 7.2.2
nk = 3*na            # number of asset grid points over distribution 
nk = na
ag =  np.linspace(amin, amax, nk)   # asset grid for distribution function

# initialization distribution function
gk = np.zeros((nk,2))

# Part 5: Computation of the
# stationary employment /unemployment
# with the help of the ergodic distribution 
print(prob)
pp1 = equivec1(pp0)   
nn0 = pp1[0] # measure of employed households in economy
print(pp1)
print(prob)

# Part 6: Initial guess K and tau
kk0 = (alpha/(1/beta-1+delta))**(1/(1-alpha))*nn0

# Compute w and r and b
w0 = wage_rate(kk0,nn0)
r0 = interest_rate(kk0, nn0)
b = w0*rep  # unemployment insurance


# Part 7:
# initialization of the value function, consumption and next-period asset
#              policy functions
# assumption: employed/unemployed consumes his income permanently 

ve = utility( (1-tau)*r*a+(1-tau)*w0) # utility of the employed with asset a
vu = utility( (1-tau)*r*a+b )    # utility of the unemployed with asset a
v =   np.c_[ve,vu] 
v = v/(1-beta)                  # agents consume their income 
copt = np.zeros((na,2))         # optimal consumption 
aopt = np.zeros((na,2))         # optimal next-period assets 

kk1 = kk0-1                  # initialization so that kk1 \ne kk0


nk0 = sum(ag<=kk0)              # asset grid point ag[nk0] is chosen as initial distribution
                                # => all households hold capital stock kk0

# outer loop over aggregate capital stock K
q = -1
while q<nq-1  and (crit1>tol or ngk<25000):    # iteration over K
    q = q+1
    w = wage_rate(kk0,nn0)
    r = interest_rate(kk0, nn0)

    kbarq[q] = kk0 # saving aggregate capital stock in iteration q
    
    # slow increment of the number of iterations over distribution
    # gk, corresponding to the number of iteration i over f_i in the textbook
    if ngk<25000:
        ngk = ngk+500
    
    # mean wealth during iteration over f_i for given w, r and b
    kt = np.zeros(ngk)

    # incremental increase in number of iterations over value functions
    nit=nit+2

    # loop over value function 
    crit=1+tol
    j = -1
    while j<=50 or (j<nit-1 and crit>tol):
        j=j+1
        
        vold = v + 0
        # prepare interpolation in value1()
        ve_polate = interpolate.interp1d(a,vold[:,0])
        vu_polate = interpolate.interp1d(a,vold[:,1])
        #print("iteration j~q: " + str([j,q])) 
        #print("kk0~kk1 " + str([kk0,kk1]) )
        #sec = (time.time() - start_time)
        #ty_res = time.gmtime(sec)
        #res = time.strftime("%H : %M : %S", ty_res)
        #print(res)
        #print("error value function: " + str(crit) )
        #print("error k: " + str(crit1) )
        #print("error distribution function: " + str(critg))
        
        # ----------------------------------------------------------------
        #
        # Step: Compute the household's decision function 
        #        Iteration of the value function
        #
        # ----------------------------------------------------------------- 
        
        # iteration over the employment status 
        # e=0 employed, e=1 unemployed 

        for e in range(2): 
            l0 = -1  # initialization of a'
                    # exploiting monotonocity of a'(a)
            # iteration over the asset grid a = a_1,...a_na
            for i in range(na):
                ainit = a[i]
                l = l0
                v0 = negvalue
                # iteration over a' to bracket the maximum of 
                # the rhs of the Bellman equation,
                # ax < bx < cx and a' is between ax and cx
                
                ax = a[0] 
                bx = a[0] 
                cx = amax
                
                while l<na-1:
                    l = l+1
                    if e==0:
                        c0 = (1+(1-tau)*r) * ainit + (1-tau)*w - a[l]
                    else:
                        c0 = (1+(1-tau)*r) * ainit + b - a[l]
                    
                    
                    if c0>0:
                        v1 = bellman(ainit,a[l],e)
                        if v1>v0:
                            v[i,e] = v1
                            if l==0:
                               ax=a[0]
                               bx=a[0] 
                               cx=a[1]
                            elif l==na-1:
                                ax=a[na-2] 
                                bx=a[na-1] 
                                cx=a[na-1]
                            else:
                                ax=a[l-1] 
                                bx=a[l] 
                                cx=a[l+1]
                            
                            v0=v1
                            l0=l-1
                        else:
                            l=na-1   # concavity of value function 
                        
                    else:
                        l=na-1
                    
                    
                if ax==bx:  # boundary optimum, ax=bx=a[1]  
                    bx = ax+eps*(a[1]-a[0])
                    if value(bx,e)<value(ax,e):
                        aopt[i,e] = a[0]
                    else:
                        aopt[i,e] = GoldenSectionMax(value1,ax,bx,cx,tol1)
                elif bx==cx:  # boundary optimum, bx=cx=a[na-1] 
                    bx = cx-eps*(a[na-1]-a[na-2])
                    if value(bx,e)<value(cx,e):
                        aopt[i,e] = a[na-1]
                    else:
                        aopt[i,e] = GoldenSectionMax(value1,ax,bx,cx,tol1)

                else:
                    aopt[i,e] = GoldenSectionMax(value1,ax,bx,cx,tol1)
                
                v[i,e] = bellman(a[i],aopt[i,e],e)

        x = abs(vold-v)
        crit = x.mean(0)  # mean of the columns, value functions of employed
                          # and unemployd
        crit = max(crit) 

    if q==0: # plotting the value and policy function
        fig, ax = plt.subplots()
        label1 = 'employed'
        label2 = 'unemployed'
        ax.plot(a,v[:,0], linewidth=2, label=label1)
        ax.plot(a,v[:,1], linewidth=2, label=label2)
        ax.legend()
        plt.show()

    # iteration over value function complete              
    copt[:,0] = (1+(1-tau)*r)*a + (1-tau)*w - aopt[:,0]
    copt[:,1] = (1+(1-tau)*r)*a + b - aopt[:,1]
    kritvq[q] = crit

    # iteration to find invariant distribution */

    kritg = 1+tol
    kconv = np.zeros(ngk+1) # convergence criterion 
    # Step: initialization of the distribution functions */
    
    # Variant 1: uniform distribution
    #    gk = np.ones((nk,2))/nk
    #    gk = gk * np.transpose(pp1)
    #
    # Variant 2: equal distribution, all hold KK0
    gk = np.zeros((nk,2))     
    gk[nk0,0] = pp1[0]
    gk[nk0,1] = pp1[1]
    
    for q1 in range(ngk):   # iteration over periods (dynamics of distribution)
         gk0 = gk+0             # distribution in period t
         gk = np.zeros((nk,2))    # distribution in period t+1
         
         for l in range(2):     # iteration over employment types in period t
             # prepare interpolation of optimal next-period asset policy
             aopt_polate = interpolate.interp1d(a,aopt[:,l]) 
             for i in range(nk):  # iteration over assets in period t
                 k0 = ag[i]
                 if k0 <= amin:
                     k1 = aopt[0,l]
                 elif k0 >= amax:
                    k1=aopt[na-1,l]
                 else:
                    k1 = aopt_polate(k0) # linear interpolation for a'(a) 
                                
                 if k1 <= amin:
                     gk[0,0] = gk[0,0] + gk0[i,l]*prob[l,0]
                     gk[0,1]= gk[0,1] + gk0[i,l]*prob[l,1]
                 elif k1 >= amax:
                     gk[nk-1,0] = gk[nk-1,0] + gk0[i,l]*prob[l,0]
                     gk[nk-1,1] = gk[nk-1,1] + gk0[i,l]*prob[l,1]
                 elif (k1>amin) and (k1<amax):
                     j = sum(ag<=k1) # k1 lies between ag[j-1] and ag[j]
                     n0 = (k1-ag[j-1]) / (ag[j]-ag[j-1])
                     gk[j,0] = gk[j,0] + n0*gk0[i,l]*prob[l,0]
                     gk[j,1] = gk[j,1] + n0*gk0[i,l]*prob[l,1]
                     gk[j-1,0] =gk[j-1,0] + (1-n0)*gk0[i,l]*prob[l,0]
                     gk[j-1,1] =gk[j-1,1] + (1-n0)*gk0[i,l]*prob[l,1]
            
         gk=gk/sum(sum(gk))
         kk1 = np.transpose(gk[:,0]+gk[:,1]) @ ag # new mean capital stock
         critg=sum(sum(abs(gk0-gk)))
         kconv[q1] = critg
         kt[q1] = kk1    

    if q==0: # plotting the value and policy function
        fig, ax = plt.subplots()
        label1 = 'employed'
        label2 = 'unemployed'
        ax.plot(ag[0:200],gk[0:200,0], linewidth=2, label=label1)
        ax.plot(ag[0:200],gk[0:200,1], linewidth=2, label=label2)
        ax.legend()
        plt.show()


        
    crit1 = abs((kk1-kk0)/kk0)   # end outer loop over K, q=1,..
    
    print("kk0~kk1: " + str([kk0,kk1]) )
    print("outer iteration q: " + str(q))
    print("crit value fucntion: " +str(crit))
    print("crit K: " +str(crit1))
    print("crit distribution: " + str(critg))
    sec = (time.time() - start_time)
    ty_res = time.gmtime(sec)
    res = time.strftime("%H : %M : %S", ty_res)
    print(res)    
    
    kk0 = psi1*kk0+(1-psi1)*kk1
    tau = b*pp1[1]/(w*nn0+r*kk0)


k1barq[q] = kk1
print("kk0~kk1 " + str([kk0,kk1]) )
sec = (time.time() - start_time)
ty_res = time.gmtime(sec)
res = time.strftime("%H : %M : %S", ty_res)
print(res)
print("error value function: " + str(crit) )
print("error k: " + str(crit1) )
print("error distribution function: " + str(critg))


plt.plot(kbarq)
plt.show()


plt.plot(kt)
plt.show()