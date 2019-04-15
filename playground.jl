using TensorOperations, LinearAlgebra
using Zygote: @adjoint, gradient
import Zygote

function f6(θ)
    RotationGate{1, Float64, typeof(X)}(X, θ).theta
end
f6(0.3)
gradient(f6, 0.3)

using Zygote:unbroadcast
@adjoint Base.broadcasted(::typeof(*), x::Array{<:Number}, y::Array{<:Number}) = x.*y,
  z̄ -> (nothing, unbroadcast(x, z̄ .* conj(y)), unbroadcast(y, z̄ .* conj(x)))
@adjoint Base.broadcasted(::typeof(*), x, y) = x.*y,
  z̄ -> (nothing, unbroadcast(x, z̄ .* conj(y)), unbroadcast(y, z̄ .* conj(x)))

abstract type A1{T} end
mutable struct A4{T} <: A1{Complex{T}}
    theta::T
    function A4{T}(theta) where T
        @show T
        new{T}(T(theta))
    end
end

gradient(x->A4{Float64}(x).theta |> abs, 0.3)
gradient(x->A4{Float64}(x).theta |> abs, 0.3)

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

@adjoint Base.conj(x) = conj(x), r̄ -> (conj(r̄),)

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)


Zygote.refresh()
@adjoint function tensorcontract(A, IA, B, IB, IC)
    y = tensorcontract(A, IA, B, IB, IC)
    # is the undesired gradient automatically ignored?
    y, _C -> (tensorcontract(_C, IC, conj(B), IB, IA), nothing, tensorcontract(conj(A), IA, _C, IC, IB), nothing, nothing)
end

using LinearAlgebra
bb = randn(3,3)
a = randn(ComplexF64, 3,3)
function gy(a)
    res=similar(a)
    mul!(res, a, a')
    LinearAlgebra.tr(res) |> real
end

g2(a)
gradient(gy, a)[1] - 2a

f(a)
f'(a) -2a

a = randn(ComplexF64, 3,3)
b = randn(ComplexF64, 3,3)
f2(a) = tensorcontract(a, (1,2), b, (1,2), ())[] |> real
gradient(f2, a)
f(a) = tensorcontract(a, (1,2), conj(a), (1,2), ())[] |> real
f8(a) = real(sum(a.*conj(a)))
f5(a) = real(LinearAlgebra.tr(a*a'))
f7(a) = real((a.*conj(a))[1])
h1(a) = real((a*conj(a)))
h2(a) = real((a.*conj(a)))
h3(a) = real(([a].*conj([a])))[]
h1'(0.3im)
h2'(0.3im)
h3'(0.3im)
gg(a) = (x = conj.(a); real(@tensor a[i,j]*x[i,j]))
x = conj.(a)

using MacroTools
using Profile
prettify(@macroexpand @tensor a[i,j]*x[i,j])

cf3(a) = TensorOperations.contract!(1, a, :N, conj(a), :N, false, fill(0.0im,()), (), (1,2), (), (1,2), (), ())[] |> real
cf3(a)
gradient(cf3, a)

#gg(a) = (x = conj(a); real(tensorcontract(a,(1,2),x,(1,2),())[]))
h(a) = real.(@tensoropt a[i,j]*a[j,k]*b[k,i])
f8(a)
f5(a)

function mytest(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

f(a)
@benchmark gg(a) seconds=1
@benchmark f(a) seconds=1
@benchmark gradient(gg,a) seconds=1
@benchmark gradient(f,a) seconds=1
c = randn(3,3)
gradient(gg,a)[1] - 2a
gradient(f,a)[1] - 2a
mytest(f, a)
gradient(f8, a)
mytest(f5, a)
mytest(f7, a)
mytest(x->imag(conj(x)./x), 0.2im)
mytest(gg, a)
mytest(h, a)
mytest(real, 2-3im)

using BenchmarkTools
a = randn(ComplexF64, 1000,1000)
@benchmark f($a) seconds=1
@benchmark gradient(f, $a) seconds=1

using Test
@test Zygote.gradient(f, a)[1] ≈ 2a
ngradient(gg, a)
gradient(gg, a)
@test Zygote.gradient(gg, a)[1] ≈ 2a
Zygote.gradient(h, a)[1]

using Zygote
struct A
    data::Float64
end
Base.:*(a::A, b::A) = A(a.data+b.data)

gradient(x->(A(x)*A(x)).data, 0.2)
gradient(x->([A(x)].*[A(x)])[].data, 0.2)
