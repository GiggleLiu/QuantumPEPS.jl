export MeasureResult, gensample, QPEPSRunTime

struct QPEPSRunTime
    circuit
    mblocks
end

struct QPEPSMachine
    config::QPEPSConfig
    runtime::QPEPSRunTime
end

function QPEPSMachine(config::QPEPSConfig)
    c = get_circuit(config)
    rt = QPEPSRunTime(c, collect_blocks(Measure, c))
    QPEPSMachine(c, rt)
end

struct MeasureResult
    nm::Int
    nbatch::Int
    nx::Int
    data::Array{Int,2}
end
MeasureResult(nm::Int, data::Array{Int, 2}) = MeasureResult(nm, size(data)..., data)

Base.getindex(m::MeasureResult, i::Int,j::Int) = readbit.(m.data[:,i], j)
Base.getindex(m::MeasureResult, ij) = getindex(m, (ij-1)%m.nx+1, (ij-1)÷m.nx+1)
Base.getindex(m::MeasureResult, ci::CartesianIndex) = getindex(m, ci.I...)
Base.size(m::MeasureResult) = (m.nx, m.nm)
Base.size(m::MeasureResult,i::Int) = i==1 ? m.nx : m.nm

"""
generate samples
"""
function gensample(c::QPEPSMachine, basis; nbatch=1024)
    c, rt = QPEPSMachine.config, QPEPSMachine.runtime
    nx = c.nrepeat+c.nv
    for m in rt.mblocks
        m.operator = repeat(c.nm, basis, 1:c.nm)
    end
    reg = zero_state(nqubits(rt.circuit); nbatch=nbatch)
    reg |> rt.circuit
    res = zeros(Int, nbatch, nx)

    for j=1:nx
        res[:,j] = rt.mblocks[j].results
    end
    return MeasureResult(c.nm, res)
end
