export QPEPSConfig, mbit, vbit, nspins, get_circuit
export Bag, setcontent!

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
        for ir=1:c.depth
            for ibit=1:c.nm
                # for each physical qubit, interact with its own bath.
                chain(nbit, pswap(nbit,mbit(c,ibit),vbit(c,ibit,iv)) for iv=1:c.nv) |> add!
                # interact neghboring physical qubits
                pswap(nbit,mbit(c,ibit),mbit(c,ibit%c.nm+1)) |> add!
                # interact neghboring virtual qubits
                chain(nbit, pswap(nbit,vbit(c,ibit,iv),vbit(c,ibit%c.nm+1,iv)) for iv=1:c.nv) |> add!
            end
        end
        # measure physical qubits
        locs = (mbit(c)...,)
        Bag(basis_rotor(Z, nbit, locs)) |> add!
        Measure(nbit; locs=locs, collapseto=0) |> add!
        if i==c.nrepeat
            for iv=1:c.nv
                locs = map(ibit->c.nm+c.nv*(ibit-1)+iv, (1:c.nm...,))
                Bag(basis_rotor(Z, nbit, locs)) |> add!
                Measure(nbit; locs=locs, collapseto = 0) |> add!
            end
        end
    end
    return circ
end
