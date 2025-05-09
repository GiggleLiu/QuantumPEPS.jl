# define the target model
using Yao, Yao.ConstGate, BitBasis
using CUDA; CUDA.allowscalar(false)
using YaoArrayRegister: u1rows!, mulrow!
using QuantumPEPS
using BenchmarkTools
using Profile

function run_profile(nx=6, ny=6; usecuda)
    nvirtual = 1
    depth = 2
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)

    reg0 = zero_state(nqubits(config); nbatch=1024)
    usecuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    dispatch!(qpeps.runtime.circuit, :random)
    energy(qpeps, model)
    @show nparameters(qpeps.runtime.circuit)

    Profile.init(delay=0.001)
    @profile energy(qpeps, model)
    display(Profile.print(mincount=10))
end

run_profile(6, 6; usecuda=true)

# REPORT
# Titan-V: 640ms
# CPU: 59s
