using PyCall
using TupleTools
using Zygote: @adjoint, gradient
@pyimport numpy
using TensorOperations: optimaltree, Power
using TensorOperations

@adjoint setdiff(args...) = setdiff(args...), _ -> nothing
@adjoint intersect(args...) = intersect(args...), _ -> nothing

"""
gradient(x->(collect(Tuple(i for i=1:3)); sum(x)), a)
gradient(x->(intersect((1,2), (2,3)); sum(x)), a)
"""

"""
find the optimal contraction tree.
"""
function TensorOperations.optimaltree(network, size_dict::Dict{Int, Int}=Dict{Int, Int}())
    unique_tokens = union(network...)
    optimaltree(network, Dict(token=>Power{:χ}(get(size_dict, token, 1),1) for token in unique_tokens))
end

function leg_analysis(IVS...)
    IALL = union(IVS...)
    II = intersect(IVS...)
    IC = setdiff(IALL, II)
    IALL, II, IC
end

asarray(x::Number) = fill(x, ())
asarray(x::AbstractArray) = x

function eincontract(args...)
    length(args) % 2 == 0 && throw(ArgumentError("Number of arguments should be odd!"))
    try
        numpy.einsum(args...) |> asarray
    catch
        err = join([arg isa AbstractArray ? "Tensor$(size(arg))" : arg for arg in args], ", ")
        throw(ArgumentError(err))
    end
end

function _eingrad_i(cdy, args, i)
    na = length(args)
    (i%2 == 0 || i==na) && return nothing
    #nargs = Tuple((j==i ? cdy : (j==i+1 ? args[end] : (j==na ? args[i+1] : arg))) for (j,arg) in enumerate(args))
    #@show nargs
    eincontract(((j==i ? cdy : (j==i+1 ? args[end] : (j==na ? args[i+1] : arg))) for (j,arg) in enumerate(args))...) |> conj
end

@adjoint function eincontract(args...)
    y = eincontract(args...)
    y, dy -> (cdy=conj(dy); Tuple(_eingrad_i(cdy, args, i) for i=1:length(args)))
end

_treecontract(tree::Int, IC::Nothing, args...) = args[2tree-1], args[2tree]
function _treecontract(tree::Int, IC, args...)
    C, IC0 = args[2tree-1], args[2tree]
    eincontract(C, IC0, IC)
end

function _treecontract(tree::Tuple, IC, args...)
    i,j = tree
    A, IA = _treecontract(i, nothing, args...)
    B, IB = _treecontract(j, nothing, args...)
    _IC = IC == nothing ? Tuple(leg_analysis(IA, IB)[3]) : IC
    eincontract(A, IA, B, IB, _IC), _IC
end

function treecontract(tree::Tuple, args...)
    length(args) % 2 == 0 && throw(ArgumentError("Number of arguments should be odd!"))
    _treecontract(tree, args[end], args...) |> first
end

#eincontract(::EinCode, )

struct EinCode{TYPE, TOPO, CODE}
end

macro EinCode_str(s)
    TYPE, TOPO, CODE = ein_decode(s)
    :(EinCode{$TYPE, $TOPO, $CODE})
end

function EinCode(s::String)
    TYPE, TOPO, CODE = ein_decode(s)
    EinCode{TYPE, TOPO, CODE}()
end

EinCode(TYPE, TOPO, CODE) = EinCode{TYPE, TOPO, CODE}()
function EinCode(CODE)
    topo = ein_code2topo(CODE)
    EinCode{TYPE, topo, ein_topo2type(topo)}()
end

function ein_str2code()
end

function ein_code2topo()
end

function ein_topo2type(topocode::Tuple)
end

"""
a einsum code is a contract graph.
"""
function is_contract(code::Tuple)
    all_indices = TupleTools.vcat(code...)
    counts = Dict{Int, Int}()
    for ind in all_indices
        counts[ind] = get(counts, ind, 0) + 1
    end
    all(isequal(2), counts |> values)
end

function is_decomposible(code::NTuple{N}) where N
    intersect(TupleTools.deleteat(code, N)...) |> isempty
end

ein_decode(s::String) = 1,2,3
