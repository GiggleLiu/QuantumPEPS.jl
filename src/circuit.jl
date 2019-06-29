export pswap, singlet_block, basis_rotor
export QPEPSConfig, mbit, vbit, nspins, get_circuit
export Bag, setcontent!
using YaoBlocks
using YaoArrayRegister: mulrow!, u1rows!

function Yao.apply!(reg::ArrayReg, pb::PutBlock{N,2,RotationGate{2,T,G}}) where {N,T,G<:SWAPGate}
    mask1 = bmask(pb.locs[1])
    mask2 = bmask(pb.locs[2])
    mask12 = mask1|mask2
    a, c, b_, d = mat(Rx(pb.content.theta))
    e = exp(-im/2*pb.content.theta)
    state = statevec(reg)
    for b in basis(reg)
        if b&mask1==0
            i = b+1
            i_ = b ⊻ mask12 + 1
            if b&mask2==mask2
                u1rows!(state, i, i_, a, b_, c, d)
            else
                mulrow!(state, i, e)
                mulrow!(state, i_, e)
            end
        end
    end
    return reg
end

mutable struct Bag{N}<:TagBlock{AbstractBlock, N}
    content::AbstractBlock{N}
end

Yao.content(bag) = bag.content
Yao.chcontent(bag::Bag, content) = Bag(content)
Yao.mat(bag::Bag) = mat(bag.content)
Yao.apply!(reg::AbstractRegister, bag::Bag) = apply!(reg, bag.content)
YaoBlocks.PreserveStyle(::Bag) = YaoBlocks.PreserveAll()
setcontent!(bag::Bag, content) = (bag.content = content; bag)

function YaoBlocks.print_annotation(io::IO, bag::Bag)
    printstyled(io, "[⊞] "; bold=true, color=:blue)
end

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

basis_rotor(::ZGate) = I2Gate()
basis_rotor(::XGate) = Ry(-0.5π)
basis_rotor(::YGate) = Rx(0.5π)

basis_rotor(basis::PauliGate, nbit, locs) = repeat(nbit, basis_rotor(basis), locs)

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
        # interact neghboring virtual qubits
        for ibit=1:c.nm
            # interact neghboring physical qubits
            chain(nbit, pswap(nbit,mbit(c,ibit),mbit(c,ibit%c.nm+1)) for ibit=1:c.nm-1) |> add!
            chain(nbit, pswap(nbit,vbit(c,ibit,iv),vbit(c,ibit%c.nm+1,iv)) for iv=1:c.nv) |> add!
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
