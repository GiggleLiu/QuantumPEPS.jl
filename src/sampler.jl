struct QPEPSRunTime
    circuit
    rotblocks
    mbasis
    mblocks
end

function chbasis!(rt::QPEPSRunTime, basis)
    for m in rt.mbasis   # m is a Bag
        setcontent!(m, chcontent(m.content, basis_rotor(basis)))  # It use chcontent in YaoBlocks, chcontent(::RotationGate, ::AbstractBlock)
    end 
    return rt
end

struct  QPEPSMachine{CT<:AbstractQPEPSConfig}
    config::CT
    runtime::QPEPSRunTime
    reg0::AbstractRegister
end

function QPEPSMachine(config::AbstractQPEPSConfig, reg0::AbstractRegister)
    c = get_circuit(config)
    rt = QPEPSRunTime(c, collect_blocks(RotationGate, c), collect_blocks(Bag, c), collect_blocks(Measure, c)) # define the runtime
    QPEPSMachine(config, rt, reg0)
end
 
struct MeasureResult{AT<:AbstractMatrix}
    nmeasure::Int   # 4
    nbatch::Int   # number of bacthes, 1024
    nx::Int       # number of outcomes in one batch, 4
    data::AT
end
MeasureResult(nmeasure::Int, data::AbstractMatrix) = MeasureResult(nmeasure, size(data)..., data) # size(data) = nbatch*nx

Base.getindex(m::MeasureResult, i::Int,j::Int) = readbit.(view(m.data,:,i), j) # view: creates a lightweight reference to this column without copying the data. readbit. return  j-th bit of all elements of the vector
Base.getindex(m::MeasureResult, ij) = getindex(m, (ij-1)%m.nx+1, (ij-1)÷m.nx+1)  # ij range from 1 to nbatch*nx
Base.getindex(m::MeasureResult, ci::CartesianIndex) = getindex(m, ci.I...)     # Q: why we get the bit of elements as index ?
Base.size(m::MeasureResult) = (m.nx, m.nmeasure)  # 4, 4
Base.size(m::MeasureResult,i::Int) = i==1 ? m.nx : m.nmeasure

"""
generate samples
"""
function gensample(qpeps::QPEPSMachine, basis)
    c, rt = qpeps.config, qpeps.runtime
    nx = c.nrepeat+nbath(c)   # 3+1 = 4
    chbasis!(rt, basis)
    copy(qpeps.reg0) |> rt.circuit    
    res = zeros(Int, nbatch(qpeps.reg0), nx)  # empty matrix to store the outcomes

    for j=1:nx
        res[:,j] = Vector(rt.mblocks[j].results)   # rt.mblocks[j]: The j-th measurement block in the runtime.
    end                                           
    return MeasureResult(nmeasure(c), res)
end
