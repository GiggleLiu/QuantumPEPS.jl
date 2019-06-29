export pswap, singlet_block
export QPEPSConfig, mbit, vbit, nspins, get_circuit

"""parametrized swap gate."""
function pswap(nbit::Int, i::Int, j::Int)
    put(nbit, (i,j)=>rot(SWAP, 0.0))
end

"""block for generating singlets."""
function singlet_block(nbit::Int, i::Int, j::Int)
    unit = chain(nbit)
    push!(unit, put(nbit, i=>chain(X, H)))
    push!(unit, control(nbit, -i, j=>X))
end

struct QPEPSConfig
    nm::Int
    nv::Int
    nrepeat::Int
    depth::Int
end

mbit(c::QPEPSConfig) = 1:c.nm
mbit(c::QPEPSConfig, ibit::Int) = mbit(c)[ibit]
vbit(c::QPEPSConfig, ibit::Int, iv::Int) = vbit(c, ibit)[iv]
vbit(c::QPEPSConfig, ibit::Int) = (c.nm+c.nv*(ibit-1)+1):(c.nm+c.nv*ibit)
Yao.nqubits(c::QPEPSConfig) = (c.nv+1)*c.nm
nspins(c::QPEPSConfig) = c.nm*c.nrepeat + c.nv*c.nm

function get_circuit(c::QPEPSConfig)
    nbit = nqubits(c)
    circ = chain(nbit)
    add!(block) = push!(circ, block)
    for i=1:c.nrepeat
        # for physical qubits, form a singlet
        chain(nbit, singlet_block(nbit, mbit(c,ibit), mbit(c,ibit+1)) for ibit=1:2:c.nm) |> add!
        if i==1
            # create singlets if it is the initial repeatition.
            for iv=1:c.nv
                chain(nbit, singlet_block(nbit, vbit(c,ibit,iv), vbit(c,ibit+1,iv)) for ibit=1:2:c.nm) |> add!
            end
        end
        # for each physical qubit, interact with its own bath.
        for ibit=1:c.nm
            chain(nbit, pswap(nbit,mbit(c,ibit),vbit(c,ibit,iv)) for iv=1:c.nv) |> add!
        end
        # interact neghboring physical qubits
        chain(nbit, pswap(nbit,mbit(c,ibit),mbit(c,ibit+1)) for ibit=1:c.nm-1) |> add!
        # interact neghboring virtual qubits
        for ibit=1:c.nm-1
            chain(nbit, pswap(nbit,vbit(c,ibit,iv),vbit(c,ibit+1,iv)) for iv=1:c.nv) |> add!
        end
        # measure physical qubits
        Measure{nbit, 4, AbstractBlock}(repeat(c.nm,Z,1:c.nm), (mbit(c)...,), 0, false) |> add!
        if i==c.nrepeat
            for iv=1:c.nv
                Measure{nbit, 4, AbstractBlock}(repeat(c.nm,Z,1:c.nm), map(ibit->c.nm+c.nv*(ibit-1)+iv, (1:c.nm...,)), 0, false) |> add!
            end
        end
    end
    return circ
end
