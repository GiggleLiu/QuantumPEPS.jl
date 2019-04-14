using FunnyTN
using FunnyTN.TensorNetworks: svdtrunc
using Yao, Yao.Blocks

function block2tensors(blk::MatrixBlock{2}; canonical=:left)
    m = reshape(permutedims(reshape(mat(blk), 2, 2, 2, 2), (1,3,2,4)), 4, 4)
    U, S, V = svdtrunc(m; tol=1e-12)
    if canonical == :left
        U, S .* V
    elseif canonical == :right
        U .* S', V
    else
        U .* sqrt.(S)', sqrt.(S) .* V
    end
    reshape(U, 2, 2, :), reshape(V, :, 2, 2)
end

@testset "parsing circuit" begin
    # method 1: direct simulation
    UA = put(3, 1=>Rx(0.5))
    C1 = control(3, 1, 2=>X)
    UB = put(3, 2=>Ry(0.2))
    C2 = control(3, 2, 3=>X)
    OZ = put(3, 3=>Z)
    c = chain(3, UA, C1, UB, C2)
    res = expect(OZ, zero_state(3) |> c)

    # method 2: tensor network contraction
    A = Matrix(mat(UA |> block))[:,1]
    B = mat(UB |> block) |> Matrix
    ZZ = mat(OZ |> block) |> Matrix

    y1 = eincontract("i,i,ij,ij,jj->", A,conj(A),B,conj(B),ZZ)
    using LinearAlgebra
    Dz = diag(ZZ)
    y2 = eincontract("i,i,ij,ij,j->", A,conj(A),B,conj(B),Dz)
    @test y1 ≈ y2
end

struct δ{N} end
δ(N::Int) = δ{N}()

# swap gate is full ranked
@show mat(SWAP)
block2tensors(SWAP)
# cnot gate is not full ranked
COPY, XOR = block2tensors(CNOT)
COPY

using QuAlgorithmZoo, MetaGraphs
c = random_diff_circuit(5, 3, pair_ring(5))

struct TNGraph
    nbits::Int
    graph::MetaGraph
end

tncontructor()

c2tn
