export MeasureResult, gensample, QPEPSRunTime, QPEPSMachine
export chbasis!

struct QPEPSRunTime
    circuit
    rotblocks
    mbasis
    mblocks
end

function chbasis!(rt::QPEPSRunTime, basis)
    for m in rt.mbasis
        setcontent!(m, chcontent(m.content, basis_rotor(basis)))
    end
    return rt
end

struct QPEPSMachine
    config::QPEPSConfig
    runtime::QPEPSRunTime
end

function QPEPSMachine(config::QPEPSConfig)
    c = get_circuit(config)
    rt = QPEPSRunTime(c, collect_blocks(RotationGate, c), collect_blocks(Bag, c), collect_blocks(Measure, c))
    QPEPSMachine(config, rt)
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
function gensample(qpeps::QPEPSMachine, basis; nbatch=1024)
    c, rt = qpeps.config, qpeps.runtime
    nx = c.nrepeat+c.nv
    chbasis!(rt, basis)
    reg = zero_state(nqubits(rt.circuit); nbatch=nbatch)
    reg |> rt.circuit
    res = zeros(Int, nbatch, nx)

    for j=1:nx
        res[:,j] = rt.mblocks[j].results
    end
    return MeasureResult(c.nm, res)
end
