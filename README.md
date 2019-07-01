# PEPS inspired quantum circuit ansatz

![](docs/images/j1j2chain44.png)

## To Install

Type `]` in a Julia REPL to enter the `pkg` mode, then type
```julia pkg
pkg> dev git@github.com:GiggleLiu/QuantumPEPS.jl.git
pkg> add Fire FileIO
```

## To Run
To run the tests
```bash
$ julia --project test/runtests.jl  # run the tests
```

To run a toy example of j1j2 model of size 4 x 4, type
```bash
$ julia --project runner.jl j1j2 4 4
```

## To Cite
```bibtex
@article{Liu2019,
    title={Variational Quantum Eigensolver with Fewer Qubits},
    author={Liu, Jin-Guo and Zhang, Yi-Hong and Wan, Yuan and Wang, Lei},
    eprint={arXiv:1902.02663},
    url={https://arxiv.org/abs/1902.02663}
}
```
