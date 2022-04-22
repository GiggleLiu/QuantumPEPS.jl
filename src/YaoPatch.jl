function Yao.apply!(reg::ArrayReg, pb::PutBlock{D,2,RotationGate{2,T,G}}) where {D,T,G<:SWAPGate}
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

mutable struct Bag{D}<:TagBlock{AbstractBlock, D}
    content::AbstractBlock{D}
end

Yao.content(bag) = bag.content
Yao.chcontent(bag::Bag, content) = Bag(content)
Yao.mat(::Type{T}, bag::Bag) where T = mat(T, bag.content)
YaoBlocks._apply!(reg::AbstractRegister, bag::Bag) = apply!(reg, bag.content)
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


