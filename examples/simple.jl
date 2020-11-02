module Simple

include("../src/GenSPPL.jl")
using .GenSPPL
using SPPL

spfunc = @spgen (debug) function model()
    X ~ SPPL.Normal(0, 2)
end

tr = simulate(spfunc, ())
display(tr)

end # module
