import numpy as np
def torus_theory(v,h,R=2.,r=1.):
    c,rho=1/6,1/8
    k1=np.cos(v)/(R+r*np.cos(v)); k2=1/r
    S=2*k1*k2
    b11=(c-rho)/6*S-rho/3*(2*k1*(k1+k2)+k1**2)
    b22=(c-rho)/6*S-rho/3*(2*k2*(k1+k2)+k2**2)
    bnn=rho/4*((k1+k2)**2+2*(k1**2+k2**2))
    dH=R*np.sin(v)/(r*(R+r*np.cos(v))**2)
    b2n=rho/2*dH
    return 1+h*h*b11,1+h*h*b22,h*h*bnn,h*h*b2n
