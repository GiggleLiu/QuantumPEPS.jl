using Yao, BitBasis
using Statistics
using Yao.ConstGate

include("AbstractModel.jl")

function pswap(nbit::Int, i::Int, j::Int)
    put(nbit, (i,j)=>rot(SWAP, 0.0))
end

"""construct a circuit for generating singlets."""
function singlet_block(nbit::Int, i::Int, j::Int)
    unit = chain(nbit)
    push!(unit, put(nbit, i=>chain(X, H)))
    push!(unit, control(nbit, -i, j=>X))
end

struct QPEPSConfig
    nm::Int
    nv::Int
    nrepeat::Int
    depth::Int
end

mbit(c::QPEPSConfig) = 1:c.nm
mbit(c::QPEPSConfig, ibit::Int) = mbit(c)[ibit]
vbit(c::QPEPSConfig, ibit::Int, iv::Int) = vbit(c, ibit)[iv]
vbit(c::QPEPSConfig, ibit::Int) = (c.nm+c.nv*(ibit-1)+1):(c.nm+c.nv*ibit)
Yao.nqubits(c::QPEPSConfig) = (c.nv+1)*c.nm
nspins(c::QPEPSConfig) = c.nm*c.nrepeat + c.nv*c.nm

using Test
@testset "QPEPSConfig" begin
    c = QPEPSConfig(4, 2, 4, 1)
    @test nqubits(c) == 12
    @test mbit(c, 3) == 3
    @test vbit(c, 4) == 11:12
    @test vbit(c, 3, 2) == 10
    @test nspins(c) == 4*4+8
    @test nparameters(get_circuit(c)) == 4*17
    @test collect_blocks(Measure, get_circuit(c))|>length == 6
end

function get_circuit(c::QPEPSConfig)
    nbit = nqubits(c)
    circ = chain(nbit)
    add!(block) = push!(circ, block)
    for i=1:c.nrepeat
        # for physical qubits, form a singlet
        chain(nbit, singlet_block(nbit, mbit(c,ibit), mbit(c,ibit+1)) for ibit=1:2:c.nm) |> add!
        if i==1
            # create singlets if it is the initial repeatition.
            for iv=1:c.nv
                chain(nbit, singlet_block(nbit, vbit(c,ibit,iv), vbit(c,ibit+1,iv)) for ibit=1:2:c.nm) |> add!
            end
        end
        # for each physical qubit, interact with its own bath.
        for ibit=1:c.nm
            chain(nbit, pswap(nbit,mbit(c,ibit),vbit(c,ibit,iv)) for iv=1:c.nv) |> add!
        end
        # interact neghboring physical qubits
        chain(nbit, pswap(nbit,mbit(c,ibit),mbit(c,ibit+1)) for ibit=1:c.nm-1) |> add!
        # interact neghboring virtual qubits
        for ibit=1:c.nm-1
            chain(nbit, pswap(nbit,vbit(c,ibit,iv),vbit(c,ibit+1,iv)) for iv=1:c.nv) |> add!
        end
        # measure physical qubits
        Measure{nbit, 4, AbstractBlock}(repeat(c.nm,Z,1:c.nm), (mbit(c)...,), 0, false) |> add!
        if i==c.nrepeat
            for iv=1:c.nv
                Measure{nbit, 4, AbstractBlock}(repeat(c.nm,Z,1:c.nm), map(ibit->c.nm+c.nv*(ibit-1)+iv, (1:c.nm...,)), 0, false) |> add!
            end
        end
    end
    return circ
end

struct QPEPSRunTime
    circuit
    mblocks
end

function QPEPSRunTime(config::QPEPSConfig)
    c = get_circuit(config)
    QPEPSRunTime(c, collect_blocks(Measure, c))
end

"""
generate samples
"""
function gensample(c::QPEPSConfig, rt::QPEPSRunTime, basis; nbatch=1024)
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

@testset "measure" begin
    mr = MeasureResult(4, [1 2 3; 4 5 6])
    @test mr.nx == 3
    @test mr.nbatch == 2
    @test mr.nm == 4
    @test mr[1, 1] == [1, 0]
    @test mr[1, 2] == [0, 0]
    @test mr[1, 3] == [0, 1]
    @test mr[1, 4] == [0, 0]

    @test mr[1, 1] == [1, 0]
    @test mr[2, 1] == [0, 1]
    @test mr[3, 1] == [1, 0]
    @test mr[CartesianIndex(3, 1)] == [1, 0]
    @test mr[3] == [1, 0]
    @test mr[10] == [0, 0]
end

function energy(c::QPEPSConfig, rt::QPEPSRunTime, model::AbstractHeisenberg; nbatch=1024)
    # measuring Z basis
    mres = gensample(circuit, rt, Z; nbatch=nbatch)
    local eng = 0.0
    for (i,j,w) in get_bonds(model)
        eng += w*mean(mres[i] .* mres[j])
    end
    eng/=4
end
