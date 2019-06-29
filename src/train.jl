export energy, train
using Flux: Optimise

function train(config, model; maxiter=200, optimizer=Optimise.ADAM(0.1), nbatch=1024, prefer_cuda=true)
    @assert nspins(config) == nspins(model)
    @assert config.nv+config.nrepeat == model.size[1]
    reg0 = zero_state(nqubits(config); nbatch=nbatch)
    HAS_CUDA && prefer_cuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    circuit = qpeps.runtime.circuit
    rotblocks = qpeps.runtime.rotblocks
    dispatch!(circuit, :random)
    println("E0 = $(energy(qpeps, model))")
    flush(stdout)

    history = Float64[]
    params = parameters(circuit)
    println("Number of parameters is $(length(params))")
    for i in 1:maxiter
        grad = map(rotblocks) do r
            r.theta += π/2
            E₊ = energy(qpeps, model)
            r.theta -= π
            E₋ = energy(qpeps, model)
            r.theta += π/2
            0.5(E₊ - E₋)
        end
        Optimise.update!(optimizer, params, grad)
        dispatch!.(rotblocks, params)
        push!(history, energy(qpeps, model))
        println("Iter $i, E = $(history[end])")
        flush(stdout)
    end
    qpeps, history
end
