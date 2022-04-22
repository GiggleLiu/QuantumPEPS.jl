module QuantumPEPS
using Yao, Yao.BitBasis
using Yao.YaoArrayRegister: mulrow!, u1rows!
using Statistics
using Yao.ConstGate
using KrylovKit: eigsolve
using Requires
using Optimisers

export pswap, singlet_block, basis_rotor, chbasis!
export QPEPSConfig, mbit, vbit, nspins, get_circuit, QMPSConfig, nbath
export Bag, setcontent!
export MeasureResult, gensample, QPEPSRunTime, QPEPSMachine
export energy, train, get_gradients
export AbstractModel, Heisenberg
export heisenberg_ij, hamiltonian, heisenberg_term, ground_state, energy, energy_exact, get_bonds, energy, heisenberg_2d, nspins
export J1J2

include("YaoPatch.jl")
include("circuit.jl")
include("sampler.jl")
include("model/AbstractModel.jl")
include("train.jl")

@init @require CuYao="b48ca7a8-dd42-11e8-2b8e-1b7706800275" include("cuda.jl")
end
