from copy import copy
from viznet import parsecircuit as _
from viznet import theme, EdgeBrush, DynamicShow, Grid, NodeBrush, CLinkBrush, QuantumCircuit, Pin, setting
from viznet.parsecircuit import C, NOT
import matplotlib.pyplot as plt
import numpy as np

def _plotinit():
    plt.axis("equal")
    plt.axis("off")

# define a grid with grid space dx = 1, and dy = 1.
GRAY = '#CCCCCC'
setting.node_setting['lw'] = 2
setting.node_setting['inner_lw'] = 2
C.zorder= 10

size = "normal"
mps = NodeBrush('tn.mps', size=size)
basic2 = NodeBrush('basic', size="small", color="w", zorder=10)
basic = NodeBrush('basic', size="small", color="k")

box = NodeBrush('box', size=size, color=GRAY, zorder=100, roundness=0.2, lw=0)
hbox = NodeBrush('box', size=(0.25,0.12), color=GRAY, zorder=100, roundness=0.06, lw=0)
obox = NodeBrush("box", ls="-", roundness=0.2, color=GRAY, zorder=-1, lw=0)
tri = NodeBrush('tn.tri', size=0.35, color="#66CCAA", rotate=-np.pi/6, lw=0)
sq = NodeBrush('tn.mpo', size=0.3, color='#EEBB44', lw=0)
sqr = NodeBrush('tn.mpo', size=0.3, color='#EE3344', lw=0)
sq2 = NodeBrush('box', size=(0.25, 0.15), color='red', zorder=103, lw=0)
dashedgate = NodeBrush('box', size="normal", ls=':')
lightgraybox = NodeBrush('box', size=(0.35, 0.4), color="#E2E2E2", lw=0, zorder=-1, roundness=0.2)
edge = EdgeBrush('-', lw=2.)
edge2 = EdgeBrush('=', lw=2.)
noline = EdgeBrush("-", color="none", lw=0)
rNC = copy(_.NC)
rNC.edgecolor='r'
doubleline = EdgeBrush("=", lw=1, color='r')

_.CROSS.size="small"
_.END.size = "small"
MRESET = copy(_.MEASURE)
MRESET.color='y'

def doublel(x, y, align='center'):
    e = edge2 >> (x, y)
    e.text("$/$", align)
    #e.text('5', 'top')
    return e

def cnot(x, y, dbl):
    p1 = (dbl.position[0], x.position[1])
    p2 = (dbl.position[0], y.position[1])
    C >> p1
    NOT >> p2
    edge >> (p1, p2)
    
def zlink(p1, p2):
    dxy = p2-p1
    t1 = p1 + [dxy[0]*0.2, 0]
    t2 = p2 - [dxy[0]*0.2, 0]
    plt.plot(*zip(p1, t1, t2, p2), color='k', lw=2)

def cornertex(s, ax, offset=(0,0), fontsize=14):
    plt.text(0.02+offset[0],0.95+offset[1],s,transform=ax.transAxes,color='k',va='top',ha='left',fontsize=fontsize, zorder=100)
