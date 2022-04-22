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

function train(config, model; maxiter=200, optimizer=Optimisers.ADAM(0.1), nbatch=1024, use_cuda=true)
    @assert nspins(config) == nspins(model)
    reg0 = zero_state(nqubits(config); nbatch=nbatch)
    use_cuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    circuit = qpeps.runtime.circuit
    rotblocks = qpeps.runtime.rotblocks
    dispatch!(circuit, :random)
    @info "E0 = $(energy(qpeps, model))"
    flush(stdout)

    history = Float64[]
    params = parameters(circuit)
    opt = Optimisers.setup(optimizer, params);
    @info "Number of parameters is $(length(params))"
    for i in 1:maxiter
        grad = get_gradients(qpeps, model)
        Optimisers.update!(opt, params, grad)
        dispatch!.(rotblocks, params)
        push!(history, energy(qpeps, model))
        @info "Iter $i, E = $(history[end])"
        flush(stdout)
    end
    qpeps, history
end
