using QuantumPEPS, Test

@testset "Demo" begin
    @test Demo.show_exact(4, 4) isa Real
    @test Demo.run_benchmark(4, 4; usecuda=false) isa Nothing
    @test Demo.gradients(4, 4; nbatch=20, maxiter=2, fix_params=false) isa Matrix
    @test Demo.j1j2peps(4, 4; nbatch=20, maxiter=5) isa Tuple
    @test Demo.j1j2mps(4, 4; nbatch=20, maxiter=5) isa Tuple
    @test Demo.j1j2mps(4, 4; nbatch=20, maxiter=5, write=true) isa Tuple
end