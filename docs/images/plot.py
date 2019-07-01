#!/usr/bin/env python
import fire
from viznet import parsecircuit as _
from theme import *
from plotlib import *

def singlet(handler, isodd, with_x=True):
    if isodd:
        handler.gate((basic, basic), (1, 0), ("S", "S"))
    else:
        handler.gate((_.CROSS, _.CROSS), (1, 0))
    

def singlet_easy(handler, i,j):
    handler.gate((basic, basic), (i, j), ("S", "S"))
        
def swap(handler, i,j):
    A, B = handler.gate((_.CROSS, _.CROSS), (i, j))
    C = basic2 >> ((A.position + B.position)/2)
    C.text(r"$\theta$", zorder=11)
    
def swap_block(handler, start, end):
    handler.x += 0.35
    for j in range(start, end-1):
        swap(handler, j,j+1)
        handler.x += 0.35
        
def init(handler, i):
    line = handler.edge
    handler.edge = noline
    handler.gate(tri, i, r"$0$", fontsize=16)
    handler.edge = line
    
def measure(handler, i, basis):
    return handler.gate(sq, i, basis)
    
def measure_reset(handler, i, basis):
    return handler.gate(sqr, i, basis)

class PLT(object):
    def j1j2(self, tp="png"):
        basis=r"$\alpha$"
        Ny = 4
        Nv = 1
        Ng = Nv+1
        num_bit = Ny*Ng
        Nx = 4
        handler = QuantumCircuit(num_bit=num_bit, locs=-np.append([0], np.cumsum(([0.8]*Nv+[1.2])*Ny)[:-1]), lw=2)
        
        with NoBoxPlt(figsize=(10,6), filename="j1j2chain.%s"%tp) as plt:
            handler.x -= 0.3
            for i in range(num_bit):
                init(handler, i)
            handler.x += 0.8
            
            for k in range(Ng):
                for i in range(0,Ny,2):
                    singlet_easy(handler, i*Ng+k,i*Ng+Ng+k)
                handler.x+=0.3
                    
            for i in range(Nx-Nv):
                if i!=0:
                    for j in range(0,Ny,2):
                        singlet_easy(handler, j*Ng,j*Ng+Ng)
                    handler.x+=0.3
 
                handler.x+=0.2
                for j in range(Nv):
                    for k in range(Ny):
                        swap(handler, k*Ng+j, k*Ng+j+1)
                    handler.x += 0.3

                handler.x+=0.2
                for k in range(Ng):
                    for j in range(0,Ny,2):
                        swap(handler, j*Ng+k, j*Ng+Ng+k)
                    handler.x+=0.4
 
                for k in range(Ng):
                    for j in range(1,Ny-1,2):
                        swap(handler, j*Ng+k, j*Ng+Ng+k)
                    handler.x+=0.4
 
                handler.x+=0.3
                for k in range(Ny):
                    if i!=Nx-Nv-1:
                        m = measure_reset(handler, k*Ng, basis)
                handler.x += 0.7

            handler.x -= 0.7
            for i in range(num_bit):
                measure(handler, i, basis)

    def j1j244(self, tp="png"):
        basis=r"$\alpha$"
        Ny = 4
        Nv = 1
        Ng = Nv+1
        num_bit = Ny*Ng
        Nx = 4
        handler = QuantumCircuit(num_bit=num_bit, locs=-np.append([0], np.cumsum(([0.8]*Nv+[1.2])*Ny)[:-1]), lw=2)
        box = NodeBrush("box", ls='--', size='small', edgecolor='#FF0000')
        
        with NoBoxPlt(figsize=(12,6), filename="j1j2chain44.%s"%tp) as plt:
            handler.x -= 0.3
            for i in range(num_bit):
                init(handler, i)
            handler.x += 0.8
            
            for k in range(Ng):
                for i in range(0,Ny,2):
                    singlet_easy(handler, i*Ng+k,i*Ng+Ng+k)
                handler.x+=0.3
                    
            for i in range(Nx-Nv):
                if i!=0:
                    for j in range(0,Ny,2):
                        singlet_easy(handler, j*Ng,j*Ng+Ng)
                    handler.x+=0.3
 
                handler.x+=0.5
                if i==0:
                    with handler.block(slice(0, num_bit-1), pad_x=0, pad_y = 0.1) as b:
                        for j in range(Nv):
                            for k in range(Ny):
                                swap(handler, k*Ng+j, k*Ng+j+1)
                            handler.x += 0.3

                        handler.x+=0.2
                        for k in range(Ng):
                            for j in range(0,Ny):
                                swap(handler, j*Ng+k, (j+1)%Ny*Ng+k)
                                handler.x+=0.4
                        handler.x-=0.4
                else:
                    for j in range(Nv):
                        for k in range(Ny):
                            swap(handler, k*Ng+j, k*Ng+j+1)
                        handler.x += 0.3

                    handler.x+=0.2
                    for k in range(Ng):
                        for j in range(0,Ny):
                            swap(handler, j*Ng+k, (j+1)%Ny*Ng+k)
                            handler.x+=0.4
                        handler.x-=0.4
                handler.x+=0.7
                b[0].text(r"$\times d$", "top")
 
                handler.x+=0.3
                for k in range(Ny):
                    if i!=Nx-Nv-1:
                        m = measure_reset(handler, k*Ng, basis)
                handler.x += 1.0

            handler.x -= 0.7
            for i in range(num_bit):
                measure(handler, i, basis)

            box >> (slice(5.5,16.5), slice(-5,-1.5))
           
fire.Fire(PLT())
