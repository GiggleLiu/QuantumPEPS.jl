export QPEPSConfig, mbit, vbit, nspins, get_circuit, QMPSConfig, nbath
export Bag, setcontent!

abstract type AbstractQPEPSConfig end
struct QPEPSConfig <: AbstractQPEPSConfig
    nm::Int
    nv::Int
    nrepeat::Int
    depth::Int
end

struct QMPSConfig <: AbstractQPEPSConfig
    nv::Int
    nrepeat::Int
    depth::Int
end

nm(c::QMPSConfig) = 1
nm(c::QPEPSConfig) = c.nm
# index physical qubits
mbit(c::AbstractQPEPSConfig) = 1:nm(c)
mbit(c::AbstractQPEPSConfig, ibit::Int) = mbit(c)[ibit]

# index virtual qubits
vbit(c::AbstractQPEPSConfig, ibit::Int, iv::Int) = vbit(c, ibit)[iv]
vbit(c::AbstractQPEPSConfig, ibit::Int) = (nm(c)+c.nv*(ibit-1)+1):(nm(c)+c.nv*ibit)
Yao.nqubits(c::AbstractQPEPSConfig) = (c.nv+1)*nm(c)

# total system spins
nspins(c::QPEPSConfig) = c.nm*c.nrepeat + c.nv*c.nm
nspins(c::QMPSConfig) = c.nrepeat + c.nv-1  # one of them is an ancilla

# number of bath qubits
nbath(c::QPEPSConfig) = c.nv
nbath(c::QMPSConfig) = c.nv-1

"""get a SU(2) symmetric PEPS ansatz."""
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

"""get a SU(2) symmetric MPS ansatz."""
function get_circuit(c::QMPSConfig)
    @assert c.nrepeat%2 == 0
    @assert c.nv%2 == 1
    nbit = nqubits(c)
    circ = chain(nbit)
    add!(block) = push!(circ, block)
    for i=1:c.nrepeat
        # for physical qubits, form a singlet
        if i%2==1
            singlet_block(nbit, 1, nbit) |> add!
        else
            swap(nbit, 1, nbit) |> add!
        end
        if i==1
            # create singlets if it is the initial repeatition.
            chain(nbit, singlet_block(nbit, vbit(c,1,iv), vbit(c,1,iv+1)) for iv=1:2:nbath(c)) |> add!
        end
        for ir=1:c.depth
            chain(nbit, pswap(nbit,iv,iv%c.nv+1) for iv=1:c.nv) |> add!
        end
        # measure physical qubits
        locs = (1,)
        Bag(basis_rotor(Z, nbit, locs)) |> add!
        Measure(nbit; locs=locs, collapseto=0) |> add!
        if i==c.nrepeat
            for iv=1:nbath(c)
                Bag(basis_rotor(Z, nbit, (1+iv,))) |> add!
                Measure(nbit; locs=(1+iv,), collapseto = 0) |> add!
            end
        end
    end
    return circ
end
