# define the target model
using Yao, Yao.ConstGate, BitBasis
using CuYao
using YaoArrayRegister: u1rows!, mulrow!
using QuantumPEPS
using CUDAnative: device!, CuDevice
using CuArrays
CuArrays.allowscalar(false)

function run_training(nx=6, ny=4)
    nv = 2
    depth = 2
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(ny, nv, nx-nv, depth)
    res = train(config, model; maxiter=200, α=0.1, nbatch=1024)
end

function show_exact(nx=6, ny=4)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    @show ground_state(model)[1]
end
