using QuantumPEPS
using Yao
using Flux

nx=4
ny=4
depth=1
nv=5
nbatch=1024
maxiter=200
J2=0.5
lr=0.1
periodic=false

model = J1J2(nx, ny; J2=J2, periodic=periodic)
config = QMPSConfig(nv, nx*ny-nv+1, depth)
optimizer = Flux.Optimise.ADAM(lr)
qpeps, history = train(config, model; maxiter=maxiter, nbatch=nbatch, optimizer=optimizer, use_cuda=false)
