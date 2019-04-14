include("eincode.jl")


############### Test Code ###################
using Test
@testset "is_contract, is_decomposible" begin
    @test is_contract(((1,2),(2,3),(1,3)))
    @test !is_contract(((1,2),(1,3), (1,4),(2,3,4)))
    @test !is_decomposible(((1,2),(2,3),(1,3)))
    @test is_decomposible(((1,2),(2,3), (3,4),(1,4)))
end

function mytest(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "einsum bp" begin
    a = randn(ComplexF64, 3,3)
    f2(a) = eincontract(a, (1,2), conj(a), (1,3), a, (1,4), (2,3,4)) |> sum |> real
    gradient(f2, a)

    @test mytest(f2, a)
    @test mytest(a->treecontract(((1,2),3), a, (1,2), a, (2,3), a, (3, 1), ())[] |> real, a)
end
