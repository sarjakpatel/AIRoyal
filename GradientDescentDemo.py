# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:16:26 2020

@author: himan
"""

import numpy as np
X=[0.5,2.5]
Y=[0.2,0.9]

def f(w,b,x): #sigmoid function with parameter w and b
    return 1.0/(1.0+np.exp(-(w*x+b)))

def error(w,b):
    err=0.0
    for x,y in zip(X,Y):
        fx=f(w,b,x)
        err+=0.5*(fx-y)**2
        
    return err
        
def grad_b(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)*(fx)*(1-fx)



def grad_w(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)*(fx)*(1-fx)*x


def do_gradient_descent():
    w,b,eta,max_epoch=1.8,0.0,1.5,1000
    for i in range(max_epoch):
        dw=0
        db=0
        for x,y in zip(X,Y):
            dw+=grad_w(w, b, x, y)
            db+=grad_b(w,b,x,y)
            
        w=w-eta*dw
        b=b-eta*db
        if(i%10==0):
            print("At",i,"iteration loss is :",str(error(w,b)))
        
        
        #print("W",w)
        #print("B",b)
        #print("DW",dw)
        #print("DB",db)
    print("Final loss is :",str(error(w, b)))
do_gradient_descent()
    
    
    