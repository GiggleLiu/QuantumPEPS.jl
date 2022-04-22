module QuantumPEPS
using Yao, BitBasis
using Statistics
using Yao.ConstGate
using KrylovKit: eigsolve
using Requires
using Optimisers

include("YaoPatch.jl")
include("circuit.jl")
include("sampler.jl")
include("model/AbstractModel.jl")
include("train.jl")

@init @require CuYao="b48ca7a8-dd42-11e8-2b8e-1b7706800275" include("cuda.jl")
end
