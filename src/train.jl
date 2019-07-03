export energy, train, get_gradients
using Flux: Optimise

function get_gradients(qpeps::QPEPSMachine, model)
    return map(qpeps.runtime.rotblocks) do r
        r.theta += π/2
        E₊ = energy(qpeps, model)
        r.theta -= π
        E₋ = energy(qpeps, model)
        r.theta += π/2
        0.5(E₊ - E₋)
    end
end

function train(config, model; maxiter=200, optimizer=Optimise.ADAM(0.1), nbatch=1024, use_cuda=true)
    @assert nspins(config) == nspins(model)
    #@assert config.nv+config.nrepeat == model.size[1]
    reg0 = zero_state(nqubits(config); nbatch=nbatch)
    use_cuda && (reg0 = reg0 |> cu)
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
        grad = get_gradients(qpeps, model)
        Optimise.update!(optimizer, params, grad)
        dispatch!.(rotblocks, params)
        push!(history, energy(qpeps, model))
        println("Iter $i, E = $(history[end])")
        flush(stdout)
    end
    qpeps, history
end
