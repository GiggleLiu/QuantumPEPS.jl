module Demo
using Yao, Yao.BitBasis
using Optimisers
using Random
using DelimitedFiles, JLD2
using ..QuantumPEPS
using CuYao: cu

function save_training(filename, qopt, loss::Vector, params::Vector)
    save(filename, "qopt", qopt, "loss", loss, "params", params)
end

"""
    j1j2peps(nx::Int=4, ny::Int=4; depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=200,
                    J2::Float64=0.5, lr::Float64=0.1,
                    periodic::Bool=false, use_cuda::Bool=false, write::Bool=false)

Faithful QPEPS traning for solving the J1-J2 hamiltonian ground state.
Returns a triple of (optimizer, history, params).

Positional Arguments
--------------------------
* `nx` and `ny` are the square lattice sizes.

Keyword Arguments
--------------------------
* `depth` is the circuit depth, decides how many entangle layers between two measurements.
* `nvirtual` is the number of virtual qubits.
* `nbatch` is the batch size, or the number of shots.
* `maxiter` is the number of optimization iterations.
* `J2` is the strength of the second nearest neighbor coupling.
* `lr` is the learning rate of the ADAM optimizer.
* `periodic` specifies the boundary condition of the lattice.
* `use_cuda` is true means uploading the code on GPU for faster computation.
* `write` is true will write training results to the data folder.
"""
function j1j2peps(nx::Int=4, ny::Int=4;
                    depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=200,
                    J2::Float64=0.5, lr::Float64=0.1,
                    periodic::Bool=false, use_cuda::Bool=false, write::Bool=false)
    Random.seed!(2)
    model = J1J2(nx, ny; J2=J2, periodic=periodic)
    config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)
    optimizer = Optimisers.ADAM(lr)
    qpeps, history = train(config, model; maxiter, nbatch, optimizer, use_cuda)
    params = parameters(qpeps.runtime.circuit)
    write && save_training(QuantumPEPS.project_relative_path("data", "j1j2-nx$nx-ny$ny-nv$nvirtual-d$depth.jld2"), optimizer, history, params)
    return optimizer, history, params
end

"""
    j1j2mps(nx::Int=4, ny::Int=4; depth::Int=3, nvirtual::Int=5,
                    nbatch::Int=1024, maxiter::Int=200,
                    J2::Float64=0.5, lr::Float64=0.1,
                    periodic::Bool=false, use_cuda::Bool=false, write::Bool=false)

Faithful QMPS traning for solving the J1-J2 hamiltonian ground state.
Returns a triple of (optimizer, history, params).

The parameters are the same as those for `j1j2peps`.
"""
function j1j2mps(nx::Int=4, ny::Int=4; depth::Int=3, nvirtual::Int=5,
                    nbatch::Int=1024, maxiter::Int=200,
                    J2::Float64=0.5, lr::Float64=0.1,
                    periodic::Bool=false, use_cuda::Bool=false, write::Bool=false)
    Random.seed!(2)
    model = J1J2(nx, ny; J2=J2, periodic=periodic)
    config = QMPSConfig(; nvirtual, depth, nrepeat=nx*ny-nvirtual+1)
    optimizer = Optimisers.ADAM(lr)
    qpeps, history = train(config, model; maxiter, nbatch, optimizer, use_cuda)
    params = parameters(qpeps.runtime.circuit)
    write && save_training(QuantumPEPS.project_relative_path("data", "j1j2-nx$nx-ny$ny-nv$nvirtual-d$depth.jld2"), optimizer, history, params)
    return optimizer, history, params
end

function gradients(nx::Int=4, ny::Int=4;
                    depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=20,
                    J2::Float64=0.5,
                    periodic::Bool=false, use_mps::Bool=false,
                    use_cuda::Bool=false, write::Bool=false, fix_params::Bool=false)
    model = J1J2(nx, ny; J2, periodic)
    if use_mps
        config = QMPSConfig(; nvirtual, depth, nrepeat=nx*ny-nvirtual+1)
    else
        config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)
    end
    Random.seed!(2)
    reg0 = zero_state(nqubits(config); nbatch=nbatch)

    use_cuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    nparams = nparameters(qpeps.runtime.circuit)
    @info "Number of parameters is $nparams"

    gradients = zeros(Float64, nparams, maxiter)
    dispatch!(qpeps.runtime.circuit, :random)
    for i=1:maxiter
        @info "Iteration $i"
        flush(stdout)
        if !fix_params
            dispatch!(qpeps.runtime.circuit, :random)
        end
        gradients[:,i] = get_gradients(qpeps, model)
    end
    write && writedlm(QuantumPEPS.project_relative_path("data", "$(fix_params ? "fixparam-gradients" : "gradients")-nx$nx-ny$ny-nv$nvirtual-d$depth-B$nbatch-iter$maxiter$(use_mps ? "mps" : "").dat", gradients))
    return gradients
end

function show_exact(nx::Int=4, ny::Int=4)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    eng = ground_state(model)[1]
    @info "The exact ground state energy is: $eng"
    return eng
end

function run_benchmark(nx::Int, ny::Int; usecuda::Bool=false, nrun::Int=10, nvirtual::Int=1, depth::Int=2, nbatch::Int=1024, transpose::Bool=false)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    config = QPEPSConfig(; nmeasure=ny, nrepeat=nx-nvirtual, nvirtual, depth)

    reg0 = zero_state(nqubits(config); nbatch=nbatch)
    transpose && (reg0 = transpose_storage(reg0))
    usecuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    dispatch!(qpeps.runtime.circuit, :random)
    energy(qpeps, model)
    @info "running $nrun times"
    @time for _=1:nrun
        energy(qpeps, model)
    end
end
end