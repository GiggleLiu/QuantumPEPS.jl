using FunnyTN.Tensors
using FunnyTN.TensorNetworks
using LightGraphs
import LightGraphs: rem_edge!, rem_vertex!, nv, ne, vertices, edges, is_directed, neighbors

# if allowing star like bond, the graph is not simple
# also, a tensor network can be a multi-graph

struct GeneralTensorNetwork{T, TT, BT<:Bond, ET} <: AbstractTN{T, TT}
    tensors::Vector{TT}
    bonds::Vector{BT}
    legmap::Matrix{ET}
end

legmap(gtn::GeneralTensorNetwork) = gtn.legmap

deleteat(a::AbstractVector, indices) = deleteat!(copy(a), indices)
function deleteat(a::AbstractMatrix, axis::Int, indices)
    Ni = length(indices)
    sizes = [size(a)...]
    sizes[axis] -= Ni
    b = similar(a, sizes...)
    inds = setdiff(1:size(a, axis), indices)
    if axis == 1
        for (i, k) in enumerate(inds)
            @inbounds b[i,:] = view(a,k,:)
        end
    else
        for (i, k) in enumerate(inds)
            @inbounds b[:,i] = view(a,:,k)
        end
    end
    b
end

nv(gtn::GeneralTensorNetwork) = size(gtn.legmap, 1)
ne(gtn::GeneralTensorNetwork) = size(gtn.legmap, 2)
#has_vertex(gtn::GeneralTensorNetwork, tensor) = tensor in gtn.tensors
#has_edge(gtn::GeneralTensorNetwork, b::Bond) = b in gtn.tensors
vertices(gtn::GeneralTensorNetwork) = gtn.tensors
edges(gtn::GeneralTensorNetwork) = gtn.bonds
is_directed(gtn::GeneralTensorNetwork) = false
function neighbors(gtn::GeneralTensorNetwork, itensor)
    edges = findall(!iszero, @inbounds view(gtn.legmap, i, :))
    nbs = Int[]
    for ie in edges
        for it in findall(!iszero, @inbounds view(gtn.legmap, :, ie))
            if (it!=itensor) && !(it in nbs)
                push!(nbs, it)
            end
        end
    end
    nbs
end

# remove edge i
function rem_edge(gtn::GeneralTensorNetwork, i::Int)
    GeneralTensorNetwork(gtn.tensors, deleteat(gtn.bonds, i), deleteat(gtn.legmap, 1, i))
end

function rem_vertex(gtn::GeneralTensorNetwork, i::Int)
    GeneralTensorNetwork(deleteat(gtn.tensors, i), gtn.bonds, deleteat(gtn.legmap, 1, i))
end

"""
contract a bond.
"""
function contract(gtn::GeneralTensorNetwork, ibond::Int)
    rem_edge(gtn.ibond)
end

function show(io::IO, g::TensorNetwork{T}) where T
    dir = is_directed(g) ? "directed" : "undirected"
    print(io, "{$(nv(g)), $(ne(g))} $dir simple $T tensor network")
end

### INTERFACE
nv(g::TensorNetwork{T}) where T = T(size(g.tensors, 1))
vertices(g::TensorNetwork{T}) where T = one(T):nv(g)
eltype(x::TensorNetwork{T}) where T = T

has_edge(g::TensorNetwork{T}, e::Bond) where T where U =
    g.weights[dst(e), src(e)] != zero(U)

# handles single-argument edge constructors such as pairs and tuples
has_edge(g::TensorNetwork{T, U}, x) where T where U = has_edge(g, edgetype(g)(x))
add_edge!(g::TensorNetwork{T, U}, x) where T where U = add_edge!(g, edgetype(g)(x))

# handles two-argument edge constructors like src,dst
has_edge(g::TensorNetwork, x, y) = has_edge(g, edgetype(g)(x, y, 0))
add_edge!(g::TensorNetwork, x, y) = add_edge!(g, edgetype(g)(x, y, 1))
add_edge!(g::TensorNetwork, x, y, z) = add_edge!(g, edgetype(g)(x, y, z))

function issubset(g::T, h::T) where T<:TensorNetwork
    (gmin, gmax) = extrema(vertices(g))
    (hmin, hmax) = extrema(vertices(h))
    return (hmin <= gmin <= gmax <= hmax) && issubset(edges(g), edges(h))
end

has_vertex(g::TensorNetwork, v::Integer) = v in vertices(g)

@doc_str """
    rem_vertex!(g::TensorNetwork, v)
Remove the vertex `v` from graph `g`. Return false if removal fails
(e.g., if vertex is not in the graph); true otherwise.
### Implementation Notes
This operation has to be performed carefully if one keeps external
data structures indexed by edges or vertices in the graph, since
internally the removal results in all vertices with indices greater than `v`
being shifted down one.
"""
function rem_vertex!(g::TensorNetwork, v::Integer)
    v in vertices(g) || return false
    n = nv(g)

    newweights = g.weights[1:nv(g) .!= v, :]
    newweights = newweights[:, 1:nv(g) .!= v]

    g.weights = newweights
    return true
end

function outneighbors(g::TensorNetwork, v::Integer)
    mat = g.weights
    return mat.rowval[mat.colptr[v]:mat.colptr[v+1]-1]
end

get_weight(g::TensorNetwork, u::Integer, v::Integer) = g.weights[v, u]

zero(g::T) where T<:TensorNetwork = T()

# TODO: manipulte SparseMatrixCSC directly
add_vertex!(g::TensorNetwork) = add_vertices!(g, 1)

copy(g::T) where T <: TensorNetwork =  T(copy(g.weights))


const SimpleWeightedGraphEdge = SimpleWeightedEdge
const SimpleWeightedDiGraphEdge = SimpleWeightedEdge
include("simpleweighteddigraph.jl")
include("simpleweightedgraph.jl")
include("overrides.jl")
include("persistence.jl")

const WGraph = SimpleWeightedGraph
const WDiGraph = SimpleWeightedDiGraph

SimpleWeightedDiGraph(g::SimpleWeightedGraph) = SimpleWeightedDiGraph(g.weights)
SimpleWeightedDiGraph{T,U}(g::SimpleWeightedGraph) where T<:Integer where U<:Real =
    SimpleWeightedDiGraph(SparseMatrixCSC{U, T}(g.weights))

SimpleWeightedGraph(g::SimpleWeightedDiGraph) = SimpleWeightedGraph(g.weights .+ g.weights')
SimpleWeightedGraph{T,U}(g::SimpleWeightedDiGraph) where T<:Integer where U<:Real =
    SimpleWeightedGraph(SparseMatrixCSC{U, T}(g.weights .+ g.weights'))


function random_gtn(n::Int, bond_dimensions)
    tensors = [randn()]
end

function demo_gtn(n::Int)
end

using Test
@testset "deleteat" begin
    a = reshape(1:16, 4,4)
    b = [1  13; 2  14; 3  15; 4  16]
    @test deleteat(a, 2, (2,3)) == b

    b = [1  5   9  13; 3  7  11  15; 4  8  12  16]
    @test deleteat(a, 1, 2) == b
end

@testset "incidence lists" begin
end
