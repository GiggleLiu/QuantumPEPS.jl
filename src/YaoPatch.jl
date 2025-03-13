mutable struct Bag{D}<:TagBlock{AbstractBlock, D}
    content::AbstractBlock{D}
end

Yao.content(bag) = bag.content    # directy get the content of the bag
Yao.chcontent(bag::Bag, content) = Bag(content)  # change the content of the bag
Yao.mat(::Type{T}, bag::Bag) where T = mat(T, bag.content)  # change content to matrix
YaoBlocks.unsafe_apply!(reg::AbstractRegister, bag::Bag) = YaoBlocks.unsafe_apply!(reg, bag.content)  # apply the content to the register
YaoBlocks.PreserveStyle(::Bag) = YaoBlocks.PreserveAll()  # preserve the style of the bag
setcontent!(bag::Bag, content) = (bag.content = content; bag)  
                                                        # |
                                                        #retuen
function YaoBlocks.print_annotation(io::IO, bag::Bag)
    printstyled(io, "[⊞] "; bold=true, color=:blue)
end
 
"""parametrized swap gate."""
function pswap(nbit::Int, i::Int, j::Int)
    put(nbit, (i,j)=>rot(SWAP, 0.0))     # first assign of theta is 0.0, then serve as parameters.
end

"""block for generating singlets."""
function singlet_block(nbit::Int, i::Int, j::Int)  # X, H and control gate to initialize the state as singlet state
    unit = chain(nbit)
    push!(unit, put(nbit, i=>chain(X, H)))
    push!(unit, control(nbit, -i, j=>X))
end

basis_rotor(::ZGate) = I2Gate()
basis_rotor(::XGate) = Ry(-0.5π)  # z-axis rotates -90 degrees around y-axis => x-axis
basis_rotor(::YGate) = Rx(0.5π)   # z-axis rotates 90 degrees around x-axis => y-axis

basis_rotor(basis::PauliGate, nbit, locs) = repeat(nbit, basis_rotor(basis), locs)


