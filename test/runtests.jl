using Test, QuantumPEPS, Yao
using LinearAlgebra, Random
using Optimisers

@testset "pswap gate" begin
    pb = QuantumPEPS.pswap(6, 2, 4)
    reg = rand_state(6)
    @test copy(reg) |> pb ≈ invoke(apply!, Tuple{ArrayReg, PutBlock}, copy(reg), pb)
    @test copy(reg) |> pb ≈ reg

    dispatch!(pb, π)
    @test copy(reg) |> pb ≈ -im*(copy(reg) |> swap(6, 2, 4))
    @test copy(reg) |> pb |> isnormalized

    dispatch!(pb, :random)
    @test copy(reg) |> pb ≈ invoke(apply!, Tuple{ArrayReg, PutBlock}, copy(reg), pb)
end

@testset "bag" begin
    bag = Bag(X)
    reg = zero_state(1)
    @test reg |> bag ≈ ArrayReg([0im, 1.0])
    @test mat(bag) == mat(X)
    @test isreflexive(bag)
    @test ishermitian(bag)
    @test isunitary(bag)

    setcontent!(bag, Z)
    print(bag)
    @test reg |> bag ≈ ArrayReg([0im, -1.0])
    @test mat(bag) == mat(Z)
end

@testset "basis rotor" begin
    mblock = Measure(1)

    reg = repeat(ArrayReg([0im,1]/sqrt(2)), 10)
    reg |> basis_rotor(Z) |> mblock
    @test mblock.results == fill(1,10)

    reg = repeat(ArrayReg([1+0im,1]/sqrt(2)), 10)
    reg |> basis_rotor(X) |> mblock
    @test mblock.results == fill(0,10)

    reg = repeat(ArrayReg([im,1.0]/sqrt(2)), 10)
    reg |> basis_rotor(Y) |> mblock
    @test mblock.results == fill(1,10)
end

@testset "QPEPSConfig" begin
    c = QPEPSConfig(; nmeasure=4, nvirtual=2, nrepeat=4, depth=1)
    @test nqubits(c) == 12
    @test mbit(c, 3) == 3
    @test vbit(c, 4) == 11:12
    @test vbit(c, 3, 2) == 10
    @test nspins(c) == 4*4+8
    @test nparameters(get_circuit(c)) == 4*20
    @test collect_blocks(Measure, get_circuit(c))|>length == 6
    @test collect_blocks(HGate, get_circuit(c))|>length == 12
end

@testset "measure" begin
    mr = MeasureResult(4, [1 2 3; 4 5 6])
    @test mr.nx == 3
    @test mr.nbatch == 2
    @test mr.nmeasure == 4
    @test mr[1, 1] == [1, 0]  # mr[i, j] calls the Base.getindex(mr, i, j) method.
    @test mr[1, 2] == [0, 0]
    @test mr[1, 3] == [0, 1]
    @test mr[1, 4] == [0, 0]

    @test mr[1, 1] == [1, 0]
    @test mr[2, 1] == [0, 1]
    @test mr[3, 1] == [1, 0]
    @test mr[CartesianIndex(3, 1)] == [1, 0]
    @test mr[3] == [1, 0]
    @test mr[10] == [0, 0]
end

@testset "qpeps machine" begin
    Random.seed!(2)
    c = QPEPSConfig(; nmeasure=4, nvirtual=2, nrepeat=4, depth=1)
    qpeps = QPEPSMachine(c, zero_state(12; nbatch=100))
    rt = qpeps.runtime
    @test length(rt.rotblocks) == 4*20
    @test length(rt.mblocks) == 6
    @test length(rt.mbasis) == 6

    @test collect_blocks(I2Gate, qpeps.runtime.circuit) |> length == 6
    chbasis!(qpeps.runtime, Y)
    @test collect_blocks(RotationGate{<:Any,<:Any,<:XGate}, qpeps.runtime.circuit) |> length == 6
    @test collect_blocks(I2Gate, qpeps.runtime.circuit) |> length == 0

    model = J1J2(6, 4; J2=0.5, periodic=false)
    @test nspins(qpeps.config) == nspins(model)
    mres = gensample(qpeps, Z)
    @test mres.nbatch == 100
    @test mres.nx == model.size[1]
    @test mres.nmeasure == model.size[2]
    @test isapprox(energy(qpeps, model), -0.75*12, atol=0.1)
    @test collect_blocks(I2Gate, qpeps.runtime.circuit) |> length == 6
end

@testset "qmps machine" begin
    Random.seed!(3)
    c = QMPSConfig(; nvirtual=5, nrepeat=12, depth=3)
    qmps = QPEPSMachine(c, zero_state(6; nbatch=100))
    rt = qmps.runtime
    @test length(rt.rotblocks) == 180
    @test length(rt.mblocks) == 16
    @test length(rt.mbasis) == 16

    @test collect_blocks(I2Gate, qmps.runtime.circuit) |> length == 16
    chbasis!(qmps.runtime, Y)
    @test collect_blocks(RotationGate{<:Any,<:Any,<:XGate}, qmps.runtime.circuit) |> length == 16
    @test collect_blocks(I2Gate, qmps.runtime.circuit) |> length == 0

    model = J1J2(4, 4; J2=0.5, periodic=false)
    @test nspins(qmps.config) == nspins(model)
    mres = gensample(qmps, Z)
    @test mres.nbatch == 100
    @test mres.nx == nspins(model)
    @test mres.nmeasure == 1
    @test isapprox(energy(qmps, model), -0.75*8, atol=0.2)
    @test collect_blocks(I2Gate, qmps.runtime.circuit) |> length == 16
end

@testset "j1j2" begin
    j1j2 = J1J2(4; periodic=false, J2=0.5)
    @test get_bonds(j1j2) == [(1, 2, 1.0),(2, 3, 1.0), (3, 4, 1.0), (1,3, 0.5), (2,4, 0.5)]
    j1j2 = J1J2(4; periodic=true, J2=0.5)
    @test get_bonds(j1j2) == [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 1, 1.0), (1,3, 0.5), (2,4, 0.5), (3,1, 0.5), (4,2, 0.5)]

    j1j2 = J1J2(3, 3; periodic=false, J2=0.5)
    @test nspins(j1j2) == 9
    prs = [1=>2, 1=>4, 2=>3, 2=>5, 2=>4, 3=>6, 3=>5, 4=>5, 4=>7, 5=>6, 5=>8, 5=>1, 5=>7, 6=>9, 6=>2, 6=>8, 7=>8, 8=>9, 8=>4, 9=>5]
    vs =  [1.0,   1,    1,    1,   0.5,  1.0,   0.5,  1.0,  1.0,  1.0, 1.0,  0.5,   0.5,  1.0,  0.5, 0.5,   1.0, 1.0,  0.5,  0.5]
    tps = [(i,j,v) for ((i,j), v) in zip(prs, vs)]
    @test sort(get_bonds(j1j2)) == sort(tps)
    j1j2 = J1J2(3, 3; periodic=true, J2=0.5)
    @test sort(get_bonds(j1j2)) == sort([(1,2,1.0), (1,4,1.0), (1,9,0.5), (1,6,0.5), (2,3,1.0), (2,5,1.0), (2,7,0.5), (2,4,0.5), (3,1,1.0), (3,6,1.0), (3,8,0.5), (3,5,0.5),
        (4,5,1.0), (4,7,1.0), (4,3,0.5), (4,9,0.5), (5,6,1.0), (5,8,1.0), (5,1,0.5), (5,7,0.5), (6,4,1.0), (6,9,1.0), (6,2,0.5), (6,8,0.5), (7,8,1.0), (7,1,1.0), (7,6,0.5), (7,3,0.5),
        (8,9,1.0), (8,2,1.0), (8,4,0.5), (8,1,0.5), (9,7,1.0), (9,3,1.0), (9,5,0.5), (9,2,0.5)])
end

@testset "train" begin
    model = J1J2(2, 4; J2=0.5, periodic=false)
    h = hamiltonian(model)
    EG = eigen(mat(h) |> Matrix).values[1]
    @test EG/nspins(model) ≈ -0.45522873432070216
end

@testset "system test" begin
    Random.seed!(2)
    nx=4
    ny=3
    depth=1
    nvirtual=3
    nbatch=200
    maxiter=50
    J2=0.5
    lr=0.1
    periodic=false

    model = J1J2(nx, ny; J2=J2, periodic=periodic)
    config = QMPSConfig(; nrepeat=nx*ny-nvirtual+1, nvirtual, depth)
    optimizer = Optimisers.ADAM(lr)
    qpeps, history = train(config, model; maxiter=maxiter, nbatch=nbatch, optimizer=optimizer, use_cuda=false)
    @test history[1] > history[end]
end

@testset "demo" begin
    include("demo.jl")
end