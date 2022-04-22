# PEPS inspired quantum circuit ansatz

[![CI](https://github.com/GiggleLiu/QuantumPEPS.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/GiggleLiu/QuantumPEPS.jl/actions/workflows/ci.yml)

To make life easier, [here](https://giggleliu.github.io/TwoQubit-VQE.html) is a simplified notebook version of MPS inspired qubit saving scheme for VQE. For a PEPS inpired ansatz solving the J1-J2 square lattice model, please checkout the following content.

![](docs/images/j1j2chain44.png)

## To Install

Type `]` in a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/index.html) to enter the `pkg` mode, then type
```julia pkg
pkg> dev https://github.com/GiggleLiu/QuantumPEPS.jl.git
```

## To Run
First, enter the directory `~/.julia/dev/QuantumPEPS/` (the default development directory of Julia) in a terminal.

To run a toy example of J1-J2 model of size 4 x 4 with J2 = 0.5, type
```bash
julia> using QuantumPEPS

julia> Demo.j1j2peps(4, 4)   # QPEPS

julia> Demo.j1j2mps(4, 4)    # QMPS
```

To get some help on input arguments, type `?` in the REPL to enter the help mode, and then type
```julia
help?> Demo.j1j2peps
  j1j2peps(nx::Int=4, ny::Int=4; depth::Int=5, nvirtual::Int=1,
                  nbatch::Int=1024, maxiter::Int=200,
                  J2::Float64=0.5, lr::Float64=0.1,
                  periodic::Bool=false, use_cuda::Bool=false, write::Bool=false)

  Faithful QPEPS traning for solving the J1-J2 hamiltonian ground state. Returns a triple of (optimizer, history, params).

  Positional Arguments
  ======================

    •  nx and ny are the square lattice sizes.

  Keyword Arguments
  ===================

    •  depth is the circuit depth, decides how many entangle layers between two measurements.

    •  nvirtual is the number of virtual qubits.

    •  nbatch is the batch size, or the number of shots.

    •  maxiter is the number of optimization iterations.

    •  J2 is the strength of the second nearest neighbor coupling.

    •  lr is the learning rate of the ADAM optimizer.

    •  periodic specifies the boundary condition of the lattice.

    •  use_cuda is true means uploading the code on GPU for faster computation.

    •  write is true will write training results to the data folder.
```

## Reference
```bibtex
@article{Liu_2019,
	doi = {10.1103/physrevresearch.1.023025},
	url = {https://doi.org/10.1103%2Fphysrevresearch.1.023025},
	year = 2019,
	month = {sep},
	publisher = {American Physical Society ({APS})},
	volume = {1},
	number = {2},
	author = {Jin-Guo Liu and Yi-Hong Zhang and Yuan Wan and Lei Wang},
	title = {Variational quantum eigensolver with fewer qubits},
	journal = {Physical Review Research}
}
```

[Download paper](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.1.023025)
