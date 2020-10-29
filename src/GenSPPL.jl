module GenSPPL

using Reexport
using PyCall

@reexport using SPPL

# Extend GFI to new DSL.
import Gen
import Gen: Selection, ChoiceMap, Trace, GenerativeFunction
import Gen: DynamicChoiceMap, EmptySelection
import Gen: get_value, has_value
import Gen: get_values_shallow, get_submaps_shallow
import Gen: get_args, get_retval, get_choices, get_score, get_gen_fn, has_argument_grads, accepts_output_grad, get_params
import Gen: select, choicemap
import Gen: simulate, generate, project, propose, assess, update, regenerate
import Gen: init_param!, accumulate_param_gradients!, choice_gradients

# ------------ Selections ------------ #

struct SPSelection{N <: NamedTuple} <: Selection
    sel::N
end
unwrap(sps::SPSelection) = sps.sel
Base.in(addr, sps::SPSelection) = haskey(unwrap(sps))
Base.getindex(sps::SPSelection, addr) = getindex(unwrap(sps), addr)
Base.isempty(sps::SPSelection, addr) = isempty(unwrap(sps), addr)

# ------------ Choice map ----------- #

struct SPChoiceMap{K <: NamedTuple} <: ChoiceMap
    chm::K
end

# ------------ Trace ------------ #

struct SPTrace{NS <: PyObject, T <: NamedTuple} <: Trace
    namespace::NS
    chm::SPChoiceMap
    marginalized::T
end

# ------------ Generative function ------------ #

struct SPFunction{N, R} <: GenerativeFunction{R, SPTrace}
    fn::Function
    arg_types::NTuple{N, Type}
    has_argument_grads::NTuple{N, Bool}
    accepts_output_grad::Bool
end

function SPFunction(func::Function, 
                    arg_types::NTuple{N, Type},
                    has_argument_grads::NTuple{N, Bool},
                    accepts_output_grad::Bool,
                    ::Type{R}) where {N, R}
end

@inline (spfunc::SPFunction)(args...) = spfunc.func(args...).sample(1)
@inline has_argument_grads(spfunc::SPFunction) = spfunc.has_argument_grads

# ------------ GFI ------------ #

function simulate(spfunc::SPFunction, args::Tuple)
    ns = spfunc.func(args...)
    m = ns.model.sample(1)
    addrs = filter(k -> k != :model, keys(ns))
    SPTrace(ns, SPChoiceMap(m), NamedTuple{addrs}([false for _ in addrs]))
end

# ----------- Macro ------------ #

macro spgen(expr)
    if expr.head == :function
        body = parse_longdef_function(expr)
    elseif expr.head == :(=)
        longdef = MacroTools.longdef(expr)
        if longdef.head == :function
            body = parse_longdef_function(longdef)
        else
            error("ParseError (@spgen): requires a longdef function definition or a shortdef function definition.")
        end
    else
        error("ParseError (@spgen): requires a longdef function definition or a shortdef function definition.")
    end

    Expr(:block, Expr(:(=), :nm, body), 
         Expr(:call, 
              GlobalRef(GenSPPL.SPFunction), 
              nm,
              Expr(:tuple),
              Expr(:tuple),
              false))
end

end # module
