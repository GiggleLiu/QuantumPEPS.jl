include("GeneralTensorNetwork.jl")
include("random_graphs.jl")
using Test

@testset "contract" begin
    g = demo_gtn()
    for i=9:-1:1
        g = contract(g, i)
    end
    @test nv(g) == 1
    @test ne(g) == 0
    @test size(g.tensors[]) == ()

    g2 = demo_gtn2()
    for i=9:-1:1
        g2 = contract(g2, i)
    end
    @test nv(g2) == 1
    @test ne(g2) == 0
    @test size(g2.tensors[]) == (2,2)
end

@testset "deleteat" begin
    a = reshape(1:16, 4,4)
    b = [1  13; 2  14; 3  15; 4  16]
    @test deleteat(a, 2, (2,3)) == b

    b = [1  5   9  13; 3  7  11  15; 4  8  12  16]
    @test deleteat(a, 1, 2) == b
end

@testset "basic simple tn" begin
    g = demo_gtn()
    @test is_simple(g)
    @test ne(g) == 9
    @test nv(g) == 7
    @test vertices(g) == 1:7
    @test vertices(g, 3) == [2,4]
    @test edges(g, 3) == [2,4,5]
    @test neighbors(g, 3) == [2,5,6]
    @test neighbors(g, (2,3)) |> sort == [1,4,5,6]
    @test occupied_legs(g, 3) |> sort == [1,2,3]
    @test dangling_legs(g, 3) == []
    @test dangling_legs(g) == repeat([Int[]], 7)
    @test occupied_legs(g) .|> sort == [[1], [1,2,3], [1,2,3], [1,2,3], [1,2], [1,2,3,4],[1,2]]
  end

@testset "basic multi-graph tn" begin
    g = demo_gtn2()
    @test !is_simple(g)
    @test ne(g) == 9
    @test nv(g) == 7
    @test vertices(g) == 1:7
    @test vertices(g, 9) == [7]
    @test edges(g, 3) |> sort == [2,4,5,6]
    @test neighbors(g, 3) == [2,5,6]
    @test occupied_legs(g, 6) |> sort == [1,3,4]
    @test dangling_legs(g, 6) == [2]
    @test dangling_legs(g) == [[2],[],[],[],[],[2],[]]
    @test occupied_legs(g) .|> sort == [[1], [1,2,3,4], [1,2,3,4], [1,2], [1,2], [1,3,4],[1,2]]
    @test filter(ie -> isloop(g, ie), g |> edges) == [9]
    # contract a selfloop
    @test dangling_legs(contract(g, 9), 7) == []
    g2 = add_vertex(g, randn(2, 2), (9,nothing))
    @test dangling_legs(g2, 8) == [2]
    @test !any(isloop.(Ref(g2), edges(g2)))
end
