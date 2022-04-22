using Comonicon
using Statistics: var, mean, std
using JLD2, DelimitedFiles

@cast function decode(nx::Int, ny::Int, depth::Int=5, nvirtual::Int=1)
    suff = "j1j2-nx$nx-ny$ny-nv$nvirtual-d$depth"
    res = load("$suff.jld2")
    writedlm("$suff-params.dat", res["params"])
    writedlm("$suff-loss.dat", res["loss"])
end

@cast function vargrad(nx::Int=4, ny::Int=4;
                    depth::Int=5, nvirtual::Int=1,
                    nbatch::Int=1024, maxiter::Int=50,
                    use_mps::Bool=false,
                    )
    gradients = readdlm("gradients-nx$nx-ny$ny-nv$nvirtual-d$depth-B$nbatch-iter$maxiter$(use_mps ? "mps" : "").dat")
    gradients = vec(gradients)
    @info "Std-Gradient = ", mean(std(gradients))

    gradients = readdlm("fixparam-gradients-nx$nx-ny$ny-nv$nvirtual-d$depth-B$nbatch-iter$maxiter$(use_mps ? "mps" : "").dat")
    @info "Std-Sampling = ", mean(std(gradients, dims=2))
end

@cast function grad_analyse(;nbatch::Int=4096)
    for use_mps in [true, false]
        for nx in [4, 6]
            @info "MPS: $use_mps, Size: $nx, Batch: $nbatch"
            maxiter = nx==4 ? 50 : 20
            depth = use_mps ? (nx==4 ? 3 : 2) : 5
            nvirtual = use_mps ? nx+1 : 1
            vargrad(nx, nx, nbatch=nbatch, maxiter=maxiter, nvirtual=nvirtual, use_mps=use_mps, depth=depth)
        end
    end
end

@main
