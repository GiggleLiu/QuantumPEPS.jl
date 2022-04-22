# please swith it off if you do not use CUDA
using Comonicon
#using Yao, Yao.ConstGate, BitBasis
#using QuantumPEPS
using Statistics: var, mean, std
using Printf
#using Flux
#using BenchmarkTools

#function decode(nx::Int, ny::Int, depth::Int=5, nvirtual::Int=1)
    #suff = QuantumPEPS.project_relative_path("data", "j1j2-nx$nx-ny$ny-nv$nvirtual-d$depth")
    #res = load("$suff.jld2")
    #writedlm("$suff-params.dat", res["params"])
    #writedlm("$suff-loss.dat", res["loss"])
#end

@cast function show_exact(nx=4, ny=4)
    model = J1J2(nx, ny; J2=0.5, periodic=false)
    @show ground_state(model)[1]
end

@cast function decode(nx::Int, ny::Int; depth::Int=5, nvirtual::Int=1)
    decode(nx, ny, depth, nvirtual)
end

@cast function vargrad(nx::Int=4, ny::Int=4;
                    depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=20,
                    use_mps::Bool=false,
                    )
    gradients = readdlm("data/gradients-nx$nx-ny$ny-nv$nvirtual-d$depth-B$nbatch-iter$maxiter$(use_mps ? "mps" : "").dat")
    gradients = vec(gradients)
    println("Std-Gradient = ", @sprintf "%.3f" mean(std(gradients)))
    #println("Mean-Abs-Gradient = ", @sprintf "%.3f" mean(abs.(gradients)))

    gradients = readdlm("data/fixparam-gradients-nx$nx-ny$ny-nv$nvirtual-d$depth-B$nbatch-iter$maxiter$(use_mps ? "mps" : "").dat")
    println("Std-Sampling = ", @sprintf "%.3f" mean(std(gradients, dims=2)))
    #println("Std-Std-Sampling = ",@sprintf "%.3f" std(std(gradients, dims=2)))
end

@cast function grad_analyse()
    for use_mps in [true, false]
        #for nbatch in [512, 1024, 2048, 4096]
        for nbatch in [4096]
            for nx in [4, 6]
                println("MPS: $use_mps, Size: $nx, Batch: $nbatch")
                maxiter = nx==4 ? 50 : 20
                depth = use_mps ? (nx==4 ? 3 : 2) : 5
                nvirtual = use_mps ? nx+1 : 1
                @show (nx, nx, nbatch, maxiter, nvirtual, use_mps, depth)
                vargrad(nx, nx, nbatch=nbatch, maxiter=maxiter, nvirtual=nvirtual, use_mps=use_mps, depth=depth)
            end
        end
    end
end

@main