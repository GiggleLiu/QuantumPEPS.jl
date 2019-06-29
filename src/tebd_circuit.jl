using Yao

function model(::Val{:peps}; nbit::Int, V::Int, B::Int=4096, nlayer::Int=5, pairs)
    c = random_circuit(1, V, nlayer, nbit-V, pairs) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 0, c, zero_state(V+1, nbatch=B), zeros(Int, nbit))
    chem
end
