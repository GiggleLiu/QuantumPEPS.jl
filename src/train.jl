export energy, train

function train(config, model; maxiter=200, α=0.1, nbatch=1024)
    @assert nspins(config) == nspins(model)
    @assert config.nv+config.nrepeat == model.size[1]
    qpeps = QPEPSMachine(config)
    dispatch!(qpeps.runtime.circuit, :random)
    for i in 1:maxiter
        for r in qpeps.runtime.rotblocks
            r.theta += π/2
            E₊ = energy(qpeps, model; nbatch=nbatch)
            r.theta -= π
            E₋ = energy(qpeps, model; nbatch=nbatch)
            r.theta += π/2
            g = 0.5(E₊ - E₋)
            r.theta -= g*α
        end
        println("Iter $i, E/N = $(energy(qpeps, model, nbatch=nbatch)/nspins(model))")
    end
    qpeps
end
