module QuantumPEPS
using Yao, Yao.BitBasis
using Yao.YaoArrayRegister: mulrow!, u1rows!
using Statistics
using Yao.ConstGate
using KrylovKit: eigsolve
using Optimisers
using CUDA

export pswap, singlet_block, basis_rotor, chbasis!
export QPEPSConfig, mbit, vbit, nspins, get_circuit, QMPSConfig, nbath
export Bag, setcontent!
export MeasureResult, gensample, QPEPSRunTime, QPEPSMachine
export energy, train, get_gradients
export AbstractModel, Heisenberg
export heisenberg_ij, hamiltonian, heisenberg_term, ground_state, energy, energy_exact, get_bonds, energy, heisenberg_2d, nspins
export J1J2, Demo

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(@__MODULE__))), xs...))

include("YaoPatch.jl")
include("circuit.jl")
include("sampler.jl")
include("model/AbstractModel.jl")
include("train.jl")

# CUDA support

include("demo.jl")

end
