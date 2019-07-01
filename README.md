# PEPS inspired quantum circuit ansatz

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
@article{Liu2019,
    title={Variational Quantum Eigensolver with Fewer Qubits},
    author={Liu, Jin-Guo and Zhang, Yi-Hong and Wan, Yuan and Wang, Lei},
    eprint={arXiv:1902.02663},
    url={https://arxiv.org/abs/1902.02663}
}
```
