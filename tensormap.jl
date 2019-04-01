using FunnyTN
using FunnyTN.TensorNetworks: svdtrunc
using Yao, Yao.Blocks
function block2tensors(blk::MatrixBlock{2})
    m = reshape(permutedims(reshape(mat(blk), 2, 2, 2, 2), (1,3,2,4)), 4, 4)
    U, S, V = svdtrunc(m; tol=1e-12)
end

# swap gate is full ranked
block2tensors(SWAP)
# cnot gate is not full ranked
block2tensors(CNOT)

using QuAlgorithmZoo, MetaGraphs
c = random_diff_circuit(5, 3, pair_ring(5))

struct TNGraph
    nbits::Int
    graph::MetaGraph
end

tncontructor()

c2tn
