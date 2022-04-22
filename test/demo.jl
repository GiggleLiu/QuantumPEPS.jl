using QuantumPEPS, Test

@testset "Demo" begin
    Demo.show_exact(4, 4)
    Demo.run_benchmark(4, 4; usecuda=false)
    Demo.gradients(4, 4; nbatch=200, maxiter=2)
    Demo.j1j2peps(4, 4; nbatch=200)
    Demo.j1j2mps(4, 4; nbatch=200)
end