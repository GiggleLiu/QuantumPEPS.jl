#eincontract(::EinCode, )

struct EinCode{TYPE, TOPO, CODE}
end

macro EinCode_str(s)
    TYPE, TOPO, CODE = ein_decode(s)
    :(EinCode{$TYPE, $TOPO, $CODE})
end

function EinCode(s::String)
    TYPE, TOPO, CODE = ein_decode(s)
    EinCode{TYPE, TOPO, CODE}()
end

EinCode(TYPE, TOPO, CODE) = EinCode{TYPE, TOPO, CODE}()
function EinCode(CODE)
    topo = ein_code2topo(CODE)
    EinCode{TYPE, topo, ein_topo2type(topo)}()
end

function ein_str2code()
end

function ein_code2topo()
end

function ein_topo2type(topocode::Tuple)
end

"""
a einsum code is a contract graph.
"""
function is_contract(code::Tuple)
    all_indices = TupleTools.vcat(code...)
    counts = Dict{Int, Int}()
    for ind in all_indices
        counts[ind] = get(counts, ind, 0) + 1
    end
    all(isequal(2), counts |> values)
end

function is_decomposible(code::NTuple{N}) where N
    intersect(TupleTools.deleteat(code, N)...) |> isempty
end

ein_decode(s::String) = 1,2,3
