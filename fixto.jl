using TensorOperations, LinearAlgebra
using Zygote: @adjoint, gradient
import Zygote

@adjoint function tensorcontract(A, IA, B, IB, IC)
    y = tensorcontract(A, IA, B, IB, IC)
    # is the undesired gradient automatically ignored?
    y, _C -> (tensorcontract(_C, IC, conj(B), IB, IA), nothing, tensorcontract(conj(A), IA, _C, IC, IB), nothing, nothing)
end

using Profile
a = randn(ComplexF64, 1000,1000)
@profile for i=1:10 gradient(gg,a) end

a = randn(ComplexF64, 3,3)
b = randn(ComplexF64, 3,3)
f(a) = tensorcontract(a, (1,2), conj(a), (1,2), ())[] |> real
h(a) = real.(@tensoropt a[i,j]*a[j,k]*b[k,i])
gg(a) = (x = conj.(a); real(@tensor a[i,j]*x[i,j]))
