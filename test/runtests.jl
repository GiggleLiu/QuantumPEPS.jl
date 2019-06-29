using Test, QuantumPEPS, Yao
using LinearAlgebra, Random

@testset "pswap gate" begin
    pb = pswap(6, 2, 4)
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
    @test mblock.results ≈ fill(1,10)

    reg = repeat(ArrayReg([1+0im,1]/sqrt(2)), 10)
    reg |> basis_rotor(X) |> mblock
    @test mblock.results ≈ fill(0,10)

    reg = repeat(ArrayReg([im,1.0]/sqrt(2)), 10)
    reg |> basis_rotor(Y) |> mblock
    @test mblock.results ≈ fill(1,10)
end

@testset "QPEPSConfig" begin
    c = QPEPSConfig(4, 2, 4, 1)
    @test nqubits(c) == 12
    @test mbit(c, 3) == 3
    @test vbit(c, 4) == 11:12
    @test vbit(c, 3, 2) == 10
    @test nspins(c) == 4*4+8
    @test nparameters(get_circuit(c)) == 4*17
    @test collect_blocks(Measure, get_circuit(c))|>length == 6
    @test collect_blocks(HGate, get_circuit(c))|>length == 12
end

@testset "measure" begin
    mr = MeasureResult(4, [1 2 3; 4 5 6])
    @test mr.nx == 3
    @test mr.nbatch == 2
    @test mr.nm == 4
    @test mr[1, 1] == [1, 0]
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
    Random.seed!(3)
    c = QPEPSConfig(4, 2, 4, 1)
    qpeps = QPEPSMachine(c)
    rt = qpeps.runtime
    @test length(rt.rotblocks) == 4*17
    @test length(rt.mblocks) == 6
    @test length(rt.mbasis) == 6

    @test collect_blocks(I2Gate, qpeps.runtime.circuit) |> length == 6
    chbasis!(qpeps.runtime, Y)
    @test collect_blocks(RotationGate{<:Any,<:Any,<:XGate}, qpeps.runtime.circuit) |> length == 6
    @test collect_blocks(I2Gate, qpeps.runtime.circuit) |> length == 0

    model = J1J2(6, 4; J2=0.5, periodic=false)
    @test nspins(qpeps.config) == nspins(model)
    mres = gensample(qpeps, Z; nbatch=100)
    @test mres.nbatch == 100
    @test mres.nx == model.size[1]
    @test mres.nm == model.size[2]
    @test isapprox(energy(qpeps, model, nbatch=100), -0.75*12, atol=0.1)
    @test collect_blocks(I2Gate, qpeps.runtime.circuit) |> length == 6
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
    @show EG/nspins(model)
end
