struct Heisenberg{D} <: AbstractHeisenberg{D}
    size::NTuple{D, Int}
    periodic::Bool
    Heisenberg(size::Int...; periodic::Bool) = new{length(size)}(size, periodic)
end

Base.size(model::Heisenberg) = model.size

heisenberg_ij(nbit::Int, i::Int, j::Int=i+1) = put(nbit, i=>X)*put(nbit, j=>X) + put(nbit, i=>Y)*put(nbit, j=>Y) + put(nbit, i=>Z)*put(nbit, j=>Z)
const heisenberg_term = repeat(2, X, 1:2) + repeat(2, Y, 1:2) + repeat(2, Z, 1:2)

function hamiltonian(model::Heisenberg)
    nbit = nspin(model)
    sum(x->heisenberg_ij(nbit, x[1], x[2]), get_bonds(model))*0.25
end

function get_bonds(model::Heisenberg{2})
    m, n = model.size
    cis = LinearIndices(model.size)
    bonds = Tuple{Int, Int, Float64}[]
    for i=1:m, j=1:n
        (i!=m || model.periodic) && push!(bonds, (cis[i,j], cis[i%m+1,j], 1.0))
        (j!=n || model.periodic) && push!(bonds, (cis[i,j], cis[i,j%n+1], 1.0))
    end
    bonds
end

function get_bonds(model::AbstractModel{1})
    nbit, = model.size
    [(i, i%nbit+1, 1.0) for i in 1:(model.periodic ? nbit : nbit-1)]
end

"""
    energy(config, model::AbstractHeisenberg; nbatch) -> Float64

Ground state energy by sampling Quantum circuit.
The hamiltonian is limited to Heisenberg and J1J2 Type.
"""
function energy(qpeps::QPEPSMachine, model::AbstractHeisenberg; nbatch=1024)
    local eng = 0.0
    for basis in [X, Y, Z]
        mres = gensample(qpeps, Z; nbatch=nbatch)
        for (i,j,w) in get_bonds(model)
            eng += w*(1-2*mean(mres[i] .⊻ mres[j]))
        end
    end
    eng/=4
end
