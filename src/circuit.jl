abstract type AbstractQPEPSConfig end
Base.@kwdef struct QPEPSConfig <: AbstractQPEPSConfig # Base.@kwdef： use default for the fields. eg: p1 = Point(x=1,y=2)
    nmeasure::Int    # 4
    nvirtual::Int    # 1
    nrepeat::Int  # means qubits repeatly used 3 
    depth::Int   # 5
end

Base.@kwdef struct QMPSConfig <: AbstractQPEPSConfig
    nvirtual::Int
    nrepeat::Int
    depth::Int
end

nmeasure(c::QMPSConfig) = 1
nmeasure(c::QPEPSConfig) = c.nmeasure
# index physical qubits
mbit(c::AbstractQPEPSConfig) = 1:nmeasure(c) # total 1:4
mbit(c::AbstractQPEPSConfig, ibit::Int) = mbit(c)[ibit] # ibit: index of the physical qubit 1,2,3,4

# index virtual qubits
vbit(c::AbstractQPEPSConfig, ibit::Int) = (nmeasure(c)+c.nvirtual*(ibit-1)+1):(nmeasure(c)+c.nvirtual*ibit) # virtual qubits of the ibit physical qubit, 
vbit(c::AbstractQPEPSConfig, ibit::Int, iv::Int) = vbit(c, ibit)[iv]   # 5,6,7,8
Yao.nqubits(c::AbstractQPEPSConfig) = (c.nvirtual+1)*nmeasure(c)  # 2*4 = 8

# total system spins
nspins(c::QPEPSConfig) = c.nmeasure*c.nrepeat + c.nvirtual*c.nmeasure # 4*3 + 1*4 = 16
nspins(c::QMPSConfig) = c.nrepeat + c.nvirtual-1  # one of them is an ancilla

# number of bath qubits
nbath(c::QPEPSConfig) = c.nvirtual
nbath(c::QMPSConfig) = c.nvirtual-1

"""get a SU(2) symmetric PEPS ansatz."""
function get_circuit(c::QPEPSConfig)
    nbit = nqubits(c)  # 8
    circ = chain(nbit)
    add!(block) = push!(circ, block) # function to add a block to the circuit

    for i=1:c.nrepeat  # there are  3 repeat blocks
        # for physical qubits, form a singlet
        chain(nbit, singlet_block(nbit, mbit(c,ibit), mbit(c,ibit+1)) for ibit=1:2:c.nmeasure) |> add!
        if i==1
            # create singlets if it is the initial repeatition.
            for iv=1:c.nvirtual
                chain(nbit, singlet_block(nbit, vbit(c,ibit,iv), vbit(c,ibit+1,iv)) for ibit=1:2:c.nmeasure) |> add!
            end
        end


        for ir=1:c.depth
            for ibit=1:c.nmeasure
                # for each physical qubit, interact with its own bath.
                chain(nbit, pswap(nbit,mbit(c,ibit),vbit(c,ibit,iv)) for iv=1:c.nvirtual) |> add!
                # interact neghboring physical qubits
                pswap(nbit,mbit(c,ibit),mbit(c,ibit%c.nmeasure+1)) |> add!
                # interact neghboring virtual qubits
                chain(nbit, pswap(nbit,vbit(c,ibit,iv),vbit(c,ibit%c.nmeasure+1,iv)) for iv=1:c.nvirtual) |> add!
            end
        end

        # measure physical qubits
        locs = (mbit(c)...,)
        Bag(basis_rotor(Z, nbit, locs)) |> add!  # messure at the Z basis
        Measure(nbit; locs=locs, resetto=0) |> add!
        if i==c.nrepeat
            for iv=1:c.nvirtual
                locs = map(ibit->c.nmeasure+c.nvirtual*(ibit-1)+iv, (1:c.nmeasure...,))
                Bag(basis_rotor(Z, nbit, locs)) |> add!
                Measure(nbit; locs=locs, resetto = 0) |> add!
            end
        end
    end
    return circ
end

"""get a SU(2) symmetric MPS ansatz."""
function get_circuit(c::QMPSConfig)
    @assert c.nrepeat%2 == 0
    @assert c.nvirtual%2 == 1
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
            chain(nbit, singlet_block(nbit, vbit(c,1,iv), vbit(c,1,iv+1)) for iv=1:2:nbath(c)) |> add! #singglet_block is in YaoPatch.jl
        end
        for ir=1:c.depth
            chain(nbit, pswap(nbit,iv,iv%c.nvirtual+1) for iv=1:c.nvirtual) |> add!
        end
        # measure physical qubits
        locs = (1,)
        Bag(basis_rotor(Z, nbit, locs)) |> add!
        Measure(nbit; locs=locs, resetto=0) |> add!
        if i==c.nrepeat
            for iv=1:nbath(c)
                Bag(basis_rotor(Z, nbit, (1+iv,))) |> add!
                Measure(nbit; locs=(1+iv,), resetto = 0) |> add!
            end
        end
    end
    return circ
end
