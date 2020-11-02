module GenSPPL

using Reexport
using PyCall
using MacroTools

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

struct SPChoiceMap{K} <: ChoiceMap
    cond::K
end

# ------------ Trace ------------ #

struct SPTrace{NS, T <: NamedTuple} <: Trace
    score::Float64
    weight::Float64
    spfunc::GenerativeFunction
    args::Tuple
    chm::SPChoiceMap
    marginalized::T
end
get_args(tr::SPTrace) = tr.args
get_retval(tr::SPTrace) = tr.chm
get_choices(tr::SPTrace) = tr.chm
get_score(tr::SPTrace) = tr.score
get_gen_fn(tr::SPTrace) = tr.spfunc
Base.getindex(tr::SPTrace, addr) = getindex(tr.chm, addr)
function project(tr::SPTrace, sps::SPSelection) end

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
    SPFunction{N, R}(func, arg_types, has_argument_grads, accepts_output_grad)
end

@inline (spfunc::SPFunction)(args...) = spfunc.func(args...).sample(1)
@inline has_argument_grads(spfunc::SPFunction) = spfunc.has_argument_grads
@inline accepts_output_grad(spfunc::SPFunction) = spfunc.accepts_output_grad

# ------------ GFI ------------ #

function simulate(spfunc::SPFunction, args::Tuple)
    ns = spfunc.fn(args...)
    s = ns.model.sample(1)[1]
    cond = foldl(&, map(collect(s)) do (k, v)
                     k << set(v)
                 end)
    score = logpdf(ns.model, cond)
    addrs = filter(k -> k != :model, keys(ns))
    SPTrace(score, 0.0, ns, SPChoiceMap(cond), NamedTuple{addrs}([false for _ in addrs]))
end
export simulate

function generate(spfunc::SPFunction, args::Tuple, choicemap::SPChoiceMap)
    ns = spfunc.fn(args...)
    conditioned_model = condition(ns.model, choicemap.cond)
    s = conditioned_model.sample(1)[1]
    s = foldl(&, map(collect(s)) do (k, v)
                     k << set(v)
                 end)
    weight = logpdf(ns.model, s)
    addrs = filter(k -> k != :model, keys(ns))
    SPTrace(score, ns, SPChoiceMap(cond), NamedTuple{addrs}([false for _ in addrs])), weight
end
export generate

function update(trace::SPTrace, args::Tuple, argdiffs::Tuple, constraints::SPChoiceMap)
    model = trace.spfunc(args...)
    conditioned_model = condition(model, constraints)
    s = sample(conditioned_model)[1]
    s = foldl(&, map(collect(s)) do (k, v)
                     k << set(v)
                 end)
    weight = logpdf(model, s) - trace.score
    SPTrace(weight, trace.score, trace.spfunc, SPChoiceMap(cond), NamedTuple{addrs}([false for _ in addrs])), weight, UndefinedChange()
end
export update

function propose(spfunc::SPFunction, args::Tuple)
    ns = spfunc.fn(args...)
    model = ns.model
    s = model.sample(1)[1]
    s = foldl(&, map(collect(s)) do (k, v)
                     k << set(v)
                 end)
    weight = logpdf(model, s)
    addrs = filter(k -> k != :model, keys(ns))
    SPChoiceMap(cond), weight, SPTrace(weight, weight, ns, SPChoiceMap(cond), NamedTuple{addrs}([false for _ in addrs])), weight
end
export propose

function assess(spfunc::SPFunction, args::Tuple, chm::SPChoiceMap)
    ns = spfunc.fn(args...)
    model = ns.model
    cond = chm.cond
    weight = logpdf(model, cond)
    weight, cond
end
export assess

# ----------- Macro ------------ #

function _spgen(expr)
    if expr.head == :function
        body = SPPL.parse_longdef_function(expr)
    elseif expr.head == :(=)
        longdef = MacroTools.longdef(expr)
        if longdef.head == :function
            body = SPPL.parse_longdef_function(longdef)
        else
            error("ParseError (@spgen): requires a longdef function definition or a shortdef function definition.")
        end
    else
        error("ParseError (@spgen): requires a longdef function definition or a shortdef function definition.")
    end

    MacroTools.@capture(expr, function fn_(args__) bd__ end)
    Expr(:block, body,
         Expr(:call, 
              GlobalRef(GenSPPL, :SPFunction), 
              fn,
              Expr(:tuple),
              Expr(:tuple),
              false,
              Any))
end

macro spgen(expr)
    new = _spgen(expr)
    new
end
macro spgen(debug, expr)
    new = _spgen(expr)
    debug == :debug && println(new)
    new
end
export @spgen

end # module
