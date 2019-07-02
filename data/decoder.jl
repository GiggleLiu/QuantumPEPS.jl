using DelimitedFiles, FileIO

function save_training(filename, qopt, loss::Vector, params::Vector)
    save(filename, "qopt", qopt, "loss", loss, "params", params)
end

function load_training(filename)
    res = load(filename)
    #res["qopt"], res["loss"], res["params"]
end

function decode(nx::Int, ny::Int, d::Int=5)
    nv = 1
    depth = 2
    suff = "j1j2-nx$nx-ny$ny-nv$nv-d$depth"
    res = load_training("$suff.jld2")
    writedlm("$suff-params.dat", res["params"])
    writedlm("$suff-loss.dat", res["loss"])
end

decode(4, 4)
