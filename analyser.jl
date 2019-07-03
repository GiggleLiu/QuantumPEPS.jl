# please swith it off if you do not use CUDA
using Fire
#using Yao, Yao.ConstGate, BitBasis
#using QuantumPEPS
using Statistics: var, mean, std
#using Flux
#using BenchmarkTools

include("data/decoder.jl")

@main function show_exact(nx=4, ny=4)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    @show ground_state(model)[1]
end

@main function decode(nx::Int, ny::Int; depth::Int=5)
    decode(nx, ny, depth)
end

@main function vargrad(nx::Int=4, ny::Int=4;
                    depth::Int=5, nv::Int=1,
                    nbatch::Int=1024, maxiter::Int=20,
                    )
    gradients = readdlm("data/gradients-nx$nx-ny$ny-nv$nv-d$depth-B$nbatch-iter$maxiter.dat")
    gradients = vec(gradients)
    println("Std-Gradient = ", mean(std(gradients)))
    println("Mean-Abs-Gradient = ", mean(abs.(gradients)))

    gradients = readdlm("data/fixparam-gradients-nx$nx-ny$ny-nv$nv-d$depth-B$nbatch-iter$maxiter.dat")
    println("Std-Sampling = ", mean(std(gradients, dims=2)))
    println("Std-Std-Sampling = ",std(std(gradients, dims=2)))
end
