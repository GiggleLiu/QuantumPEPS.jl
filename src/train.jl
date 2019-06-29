export energy, train

function energy(c::QPEPSConfig, model::AbstractHeisenberg; nbatch=1024)
    # measuring Z basis
    mres = gensample(circuit, rt, Z; nbatch=nbatch)
    local eng = 0.0
    for (i,j,w) in get_bonds(model)
        eng += w*mean(mres[i] .* mres[j])
    end
    eng/=4
end

function train(circuit, model; maxiter=200, α=0.1, nbatch=1024)
    rots = collect(RotationGate, circuit)
    for i in 1:maxiter
        for r in rots
            r.theta += π/2
            E₊ = energy(circuit, model; nbatch=nbatch)
            r.theta -= π
            E₋ = energy(circuit, model; nbatch=nbatch)
            r.theta += π/2
            g = 0.5(E₊ - E₋)
            r.theta -= g*α
        end
        println("Iter $i, E/N = $(energy(circuit, model, nbatch=nbatch)/model.length)")
    end
    circuit
end
