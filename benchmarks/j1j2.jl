# define the target model
using Yao, Yao.ConstGate, BitBasis
using CUDA; CUDA.allowscalar(false)
using YaoArrayRegister: u1rows!, mulrow!
using QuantumPEPS
using BenchmarkTools

function run_benchmark(nx=6, ny=4; usecuda, nvirtual=2, depth=1)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)

    reg0 = zero_state(nqubits(config); nbatch=1024)
    usecuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    dispatch!(qpeps.runtime.circuit, :random)
    display(@benchmark energy($qpeps, $model))
end

#run_benchmark(;usecuda=false)
#run_benchmark(;usecuda=true)
run_benchmark(6, 6; usecuda=true, nvirtual=1, depth=2)

# REPORT
# Titan-V: 640ms
# CPU: 59s
