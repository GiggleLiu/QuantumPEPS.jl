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
    if !(args[1] isa String) && is_contract((args[2:2:end]..., args[end]))
        return tensorcontract(args...)
    end
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

function _treecontract(tree::Union{Tuple, Vector}, IC, args...)
    i,j = tree
    A, IA = _treecontract(i, nothing, args...)
    B, IB = _treecontract(j, nothing, args...)
    _IC = IC == nothing ? Tuple(leg_analysis(IA, IB)[3]) : IC
    eincontract(A, IA, B, IB, _IC), _IC
end

function treecontract(tree::Union{Tuple, Vector}, args...)
    length(args) % 2 == 0 && throw(ArgumentError("Number of arguments should be odd!"))
    _treecontract(tree, args[end], args...) |> first
end

nograd(f, args...) = f(args...)
@adjoint nograd(f, args...) = f(args...), _ -> nothing
@adjoint optimaltree(args...) = optimaltree(args...), _ -> nothing

#=
macro nograd(expr)
    adjexpr = :(@adjoint optimaltree(args...) = optimaltree(args...), _ -> nothing
    ex = :(expr; )
end
=#

function _args2network(args, nt)
    network = Vector{Vector{Int}}(undef, nt)
    for i = 1:nt
        network[i] = [args[2i]...]
    end
    network
end

@adjoint _args2network(args...) = _args2network(args...), _ -> nothing

@generated function optcontract(args...)
    na = length(args)
    nt = na÷2
    na%2 == 0 && throw(ArgumentError("number of arguments must be odd, output indices must be specified."))
    for i = 1:na-1
        if i%2 == 0
            args[i] <: NTuple{<:Any, <:Integer} || throw(ArgumentError("$i-th argument should be a tuple."))
        else
            args[i] <: AbstractArray || throw(ArgumentError("$i-th argument type should be an array."))
        end
    end

    quote
        network = _args2network(args, $nt)
        # size check
        size_dict = Dict{Int, Int}()
        @inbounds for i = 1:$nt
            legs = network[i]
            ts = args[2i-1]
            for (N, leg) in zip(size(ts), legs)
                if haskey(size_dict, leg)
                    size_dict[leg] == N || throw(DimensionMismatch("size of contraction leg $leg not match."))
                else
                    size_dict[leg] = N
                end
            end
        end
        print("finding optimal tree ...")
        tree, cost = optimaltree(network, size_dict)
        @show tree, cost
        treecontract(tree, args...)
    end
end
