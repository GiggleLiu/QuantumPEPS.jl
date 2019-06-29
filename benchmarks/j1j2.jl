# define the target model
using Yao, Yao.ConstGate, BitBasis
using CuYao
using YaoArrayRegister: u1rows!, mulrow!
using QuantumPEPS
using CUDAnative: device!, CuDevice
using CuArrays
using BenchmarkTools

device!(CuDevice(3))
CuArrays.allowscalar(false)

function run_benchmark(nx=6, ny=4; usecuda)
    nv = 2
    depth = 1
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(ny, nv, nx-nv, depth)

    reg0 = zero_state(nqubits(config); nbatch=1024)
    usecuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    dispatch!(qpeps.runtime.circuit, :random)
    display(@benchmark energy($qpeps, $model))
end

run_benchmark(;usecuda=false)
run_benchmark(;usecuda=true)

# REPORT
# Titan-V: 640ms
# CPU: 59s
