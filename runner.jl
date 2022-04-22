# please swith it off if you do not use CUDA
const USE_CUDA = true

if USE_CUDA
    using CUDAnative: device!, CuDevice
    device!(CuDevice(0))   # change the device number here!
    using CuArrays
    CuArrays.allowscalar(false)
    using CuYao
end

using Comonicon
using Yao, Yao.ConstGate, BitBasis
using QuantumPEPS
using Optimisers
using BenchmarkTools, Random

include("data/decoder.jl")

@cast function j1j2(nx::Int=4, ny::Int=4;
                    depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=200,
                    J2::Float64=0.5, lr::Float64=0.1,
                    periodic::Bool=false)
    Random.seed!(2)
    model = J1J2(nx, ny; J2=J2, periodic=periodic)
    config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)
    optimizer = Optimisers.ADAM(lr)
    qpeps, history = train(config, model; maxiter, nbatch, optimizer, use_cuda=USE_CUDA)
    params = parameters(qpeps.runtime.circuit)
    save_training("data/j1j2-nx$nx-ny$ny-nv$nvirtual-d$depth.jld2", optimizer, history, params)
end

@cast function j1j2mps(nx::Int=4, ny::Int=4;
                    depth::Int=3, nvirtual::Int=5,
                    nbatch::Int=1024, maxiter::Int=200,
                    J2::Float64=0.5, lr::Float64=0.1,
                    periodic::Bool=false)
    Random.seed!(2)
    model = J1J2(nx, ny; J2=J2, periodic=periodic)
    config = QMPSConfig(; nvirtual, depth, nrepeat=nx*ny-nvirtual+1)
    optimizer = Optimisers.ADAM(lr)
    qpeps, history = train(config, model; maxiter=maxiter, nbatch=nbatch, optimizer=optimizer, use_cuda=USE_CUDA)
    params = parameters(qpeps.runtime.circuit)
    save_training("data/j1j2-nx$nx-ny$ny-nv$nvirtual-d$depth.jld2", optimizer, history, params)
end

@cast function gradients(nx::Int=4, ny::Int=4;
                    depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=20,
                    J2::Float64=0.5,
                    periodic::Bool=false, use_mps::Bool=false)
    model = J1J2(nx, ny; J2, periodic)
    if use_mps
        config = QMPSConfig(; nvirtual, depth, nrepeat=nx*ny-nvirtual+1)
    else
        config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)
    end
    Random.seed!(2)
    reg0 = zero_state(nqubits(config); nbatch=nbatch)

    USE_CUDA && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    nparams = nparameters(qpeps.runtime.circuit)
    println("Number of parameters is $nparams")

    for fix_params in [true, false]
        gradients = zeros(Float64, nparams, maxiter)
        dispatch!(qpeps.runtime.circuit, :random)
        for i=1:maxiter
            println("Iteration $i")
            flush(stdout)
            if !fix_params
                dispatch!(qpeps.runtime.circuit, :random)
            end
            gradients[:,i] = get_gradients(qpeps, model)
        end
        writedlm("data/$(fix_params ? "fixparam-gradients" : "gradients")-nx$nx-ny$ny-nv$nvirtual-d$depth-B$nbatch-iter$maxiter$(use_mps ? "mps" : "").dat", gradients)
    end
end

@cast function show_exact(nx=4, ny=4)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    @show ground_state(model)[1]
end

@cast function run_benchmark(usecuda)
    nvirtual = 1
    depth = 2
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)

    reg0 = zero_state(nqubits(config); nbatch=1024)
    usecuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    dispatch!(qpeps.runtime.circuit, :random)
    display(@benchmark energy($qpeps, $model))
end

@main