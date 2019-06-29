export energy, train

function train(config, model; maxiter=200, α=0.1, nbatch=1024, prefer_cuda=true)
    @assert nspins(config) == nspins(model)
    @assert config.nv+config.nrepeat == model.size[1]
    reg0 = zero_state(nqubits(config); nbatch=nbatch)
    HAS_CUDA && prefer_cuda && (reg0 = reg0 |> cu)
    qpeps = QPEPSMachine(config, reg0)
    dispatch!(qpeps.runtime.circuit, :random)
    println("E0 = $(energy(qpeps, model))")
    for i in 1:maxiter
        for r in qpeps.runtime.rotblocks
            r.theta += π/2
            E₊ = energy(qpeps, model)
            r.theta -= π
            E₋ = energy(qpeps, model)
            r.theta += π/2
            g = 0.5(E₊ - E₋)
            r.theta -= g*α
        end
        println("Iter $i, E = $(energy(qpeps, model))")
    end
    qpeps
end
