# define the target model
using Yao, Yao.ConstGate, BitBasis
using YaoArrayRegister: u1rows!, mulrow!
using QuantumPEPS

function run_training(nx=6, ny=4)
    nv = 2
    depth = 1
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(ny, nv, nx-nv, depth)
    qpeps = QPEPSMachine(config)
    samples = gensample(qpeps, Z; nbatch=100)

    @show energy(qpeps, model; nbatch=100)
    @show ground_state(model)
    res = train(config, model; maxiter=200, α=0.1, nbatch=1024)
end
