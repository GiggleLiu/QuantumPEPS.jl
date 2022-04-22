# PEPS inspired quantum circuit ansatz

To make life easier, [here](https://giggleliu.github.io/TwoQubit-VQE.html) is a simplified notebook version of MPS inspired qubit saving scheme for VQE. For a PEPS inpired ansatz solving the J1-J2 square lattice model, please checkout the following content.

![](docs/images/j1j2chain44.png)

## To Install

Type `]` in a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/index.html) to enter the `pkg` mode, then type
```julia pkg
pkg> add Fire FileIO
pkg> dev git@github.com:QuantumBFS/CuYao.jl.git
pkg> dev git@github.com:GiggleLiu/QuantumPEPS.jl.git
```
`CuYao` is for [CUDA](https://en.wikipedia.org/wiki/CUDA) support, do not install it for a machine without CUDA support.

## To Run
First, enter the directory `~/.julia/dev/QuantumPEPS/` (the default development directory of Julia) in a terminal.

To run tests, type
```bash
$ julia --project test/runtests.jl  # run the tests
```

To run a toy example of j1j2 model of size 4 x 4 with J2 = 0.5, type
```bash
$ julia --project runner.jl j1j2 4 4
```

Note: please turn of the CUDA swith in `runner.jl` if `CuYao` is not installed.

## To Cite
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
