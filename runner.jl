using CUDAnative: device!, CuDevice
device!(CuDevice(4))
using CuArrays
CuArrays.allowscalar(false)

using Fire
using Yao, Yao.ConstGate, BitBasis
using CuYao
using YaoArrayRegister: u1rows!, mulrow!
using QuantumPEPS
using Flux
using BenchmarkTools

include("data/decoder.jl")

@main function j1j2(nx::Int=4, ny::Int=4)
    nv = 1
    depth = 3
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(ny, nv, nx-nv, depth)
    optimizer = Flux.Optimise.ADAM(0.1)
    qpeps, history = train(config, model; maxiter=200, nbatch=1024, optimizer=optimizer)
    params = parameters(qpeps.runtime.circuit)
    save_training("data/j1j2-nx$nx-ny$ny-nv$nv-d$depth.jld2", optimizer, history, params)
end

@main function show_exact(nx=4, ny=4)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    @show ground_state(model)[1]
end

@main function run_benchmark(usecuda)
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
