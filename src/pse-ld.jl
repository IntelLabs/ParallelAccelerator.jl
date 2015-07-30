module LD

import Base.LinAlg: BlasInt

#require("intel-pse.jl")
#importall IntelPSE
import ..getPackageRoot
import ..Pert

# This controls the debug print level.  0 prints nothing.  At the moment, 2 prints everything.
DEBUG_LVL=0

function set_debug_level(x)
    global DEBUG_LVL = x
end

# A debug print routine.
function dprint(level,msgs...)
    if(DEBUG_LVL >= level)
        print(msgs...)
    end 
end

# A debug print routine.
function dprintln(level,msgs...)
    if(DEBUG_LVL >= level)
        println(msgs...)
    end 
end

type LDState
  defs  :: Array{Any, 1} # (Symbol, Type, Flag)
  exprs :: Array{Any, 1}
end

#if isdefined(LinAlg.BlasChar)
#  import Base.LinAlg: BlasChar
#else
#if !isdefined(LinAlg.BlasChar)
typealias BlasChar Char
#end
#end

empty_state() = LDState(Array(Any, 0), Array(Any, 0))
emit_expr(state, expr) = push!(state.exprs, expr)
function new_var(state, var, typ, flag) 
  push!(state.defs, (var, typ, flag))
  return var
end

call_dict = Dict{String,Symbol}("LAPACKE_dlaswp" => :new_dlaswp,   "LAPACKE_dgetrf" => :new_dgetrf,   "cblas_dtrsm" => :new_dtrsm,    "cblas_dgemm" => :new_dgemm,    "LAPACKE_dpotrf" => :new_dpotrf,   "cblas_dsyrk" => :new_dsyrk, "LAPACKE_dgeqr2" => :new_dgeqr2, "LAPACKE_dlarft" => :new_dlarft, "LAPACKE_dlarfb"  => :new_dlarfb )
lib_dict  = Dict{String,Symbol}("LAPACKE_dlaswp" => :libblas_name, "LAPACKE_dgetrf" => :libblas_name, "cblas_dtrsm" => :libblas_name, "cblas_dgemm" => :libblas_name, "LAPACKE_dpotrf" => :libblas_name, "cblas_dsyrk" => :libblas_name, "LAPACKE_dgeqr2" => :libblas_name, "LAPACKE_dlarft" => :libblas_name, "LAPACKE_dlarfb" => :libblas_name)

function check_ccall(state, expr)
  if(isa(expr, Expr) && is(expr.head, :call)) && isa(expr.args[1], TopNode) && is(expr.args[1].name, :ccall)
    dprintln(2,"got ccall: ", expr)
    args = expr.args
    if isa(args[2], Expr) && is(args[2].head, :call1) 
      dprintln(2,"length: ", length(args[2].args), " ", args[2].args[2], " ", args[2].args[3])
      dprintln(2,"args[5:end]: ", args[5:end])
      if length(args[2].args) == 3 && haskey(call_dict, args[2].args[2]) && is(args[2].args[3], lib_dict[args[2].args[2]])
         matched_call = call_dict[args[2].args[2]]
         dprintln(2, "found ", matched_call, " call!")
         real_args = Array(Any, 0)
         push!(real_args, Expr(:call, TopNode(:getfield), Expr(:call, TopNode(:getfield), IntelPSE, QuoteNode(:LD)), QuoteNode(matched_call)))
         for i = 5:2:length(args)		 
          arg = args[i] 
           if isa(arg, Expr) && is(arg.head, :&) 
             arg = arg.args[1]
           end
           push!(real_args, arg)
         end
         ld_var_typ = Ptr{Void}
         ld_var = new_var(state, gensym(), ld_var_typ, 18)
         ld_node = SymbolNode(ld_var, ld_var_typ)
         expr.args = real_args
         expr.typ  = ld_var_typ
         expr = Expr(:(=), ld_var, expr)
         expr.typ = ld_var_typ
         emit_expr(state, expr)
	 
         #expr = Expr(:call, :println, Expr(:call, TopNode(:getfield), Base, QuoteNode(:STDOUT)),args[2].args[2])
         #emit_expr(state, expr)
         #expr = Expr(:call, :println, Expr(:call, TopNode(:getfield), Base, QuoteNode(:STDOUT)),"\nnew_ returns: ", ld_node)
         #emit_expr(state, expr)
	 
         expr = Expr(:call, Expr(:call, TopNode(:getfield), Expr(:call, TopNode(:getfield), IntelPSE, QuoteNode(:LD)), QuoteNode(:insert_task)), 
                     ld_node, IntelPSE.client_intel_task_graph ? 3 : IntelPSE.client_intel_pse_mode)
         emit_expr(state, expr)
         return nothing
      end
    end
  end
  return false
end

function from_expr(state, expr)
  if isa(expr, Expr)
  dprintln(2,"from_expr: ", expr.head)
    expr_ = check_ccall(state, expr)
    if is(expr_, false)
      for i = 1:length(expr.args)
        expr.args[i] = from_expr(state, expr.args[i])
      end
    else
      expr = expr_
    end
  end
  return expr
end

function Optimize(ast, call_sig_arg_tuple, call_sig_args)
  return decompose(ast)
end

# Decompose ccalls among the expressions in the body of a lambda.
# Modification is made in place.
function decompose(ast)
  julia_root   = ENV("JULIA_ROOT")
  # Strip trailing /
  len_root     = endof(julia_root)
  if(julia_root[len_root] == '/')
    julia_root = julia_root[1:len_root-1]
  end
  # LD mode, pert_init with double buffer
  Pert.pert_init(julia_root, true)
  if isa(ast, LambdaStaticData)
      ast = uncompressed_ast(ast)
  end
  dprintln(2,"decompose: ", typeof(ast))
  assert(isa(ast, Expr) && is(ast.head, :lambda))
  body = ast.args[3]
  assert(isa(body, Expr) && is(body.head, :body))
  state = empty_state()
  for i = 1:length(body.args)
    expr = from_expr(state, body.args[i])
    emit_expr(state, expr)
  end
  body.args = state.exprs
  for (var, typ, flag) in state.defs
    push!(ast.args[2][1], var)
    push!(ast.args[2][2], Any[var, typ, flag])
  end
  dprintln(3, "after decomposition ast = ", ast)
  return ast
end


function __init__()
  libpath = string(getPackageRoot(), "src/intel-runtime/lib/libintel-runtime.so")
  @eval begin

    function divisible(obj)
      ccall((:divisible, $(libpath)), BlasInt, (Ptr{Void},), obj)
    end  

    function best_tiles(obj, proc_ID, tiles)
      ccall((:best_titles, $(libpath)), BlasInt, (Ptr{Void}, BlasInt, Ptr{Void}),
            obj, proc_ID, tiles)
    end

    function compute_bound(obj)
      ccall((:compute_bound, $(libpath)), BlasInt, (Ptr{Void},), obj)
    end  

    function coproc_biased(obj)
      ccall((:coproc_biased, $(libpath)), BlasInt, (Ptr{Void},), obj)
    end  

    function static_sched_preferred(obj)
      ccall((:static_sched_preferred, $(libpath)), BlasInt, (Ptr{Void},), obj)
    end  

    function cpu_perf(obj, num_threads)
      ccall((:cpu_perf, $(libpath)), BlasInt, (Ptr{Void}, BlasInt), obj, num_threads)
    end

    function coproc_perf(obj, num_threads)
      ccall((:coproc_perf, $(libpath)), BlasInt, (Ptr{Void},BlasInt), obj, num_threads)
    end

    function cpu_coproc_communication_time(obj, size)
      ccall((:cpu_coproc_communication_time, $(libpath)), BlasInt, (Ptr{Void},BlasInt), obj, size)
    end

    function best_chunks(obj, chunks)
      ccall((:best_chunks, $(libpath)), BlasInt, (Ptr{Void}, Ptr{Void}), obj, chunks)
    end

    function reusable(obj, p_chunk, p_dimension)
      ccall((:reusable, $(libpath)), BlasInt, (Ptr{Void}, Ptr(Void), Ptr{Void}), obj, p_chunk, p_dimension)
    end

    function SWP_non_pipelined_data(obj, data)
      ccall((:SWP_non_pipelined_data, $(libpath)), BlasInt, (Ptr{Void}, Ptr{Void}), obj, data)
    end

    function SWP_pipelined_data(obj, data)
      ccall((:SWP_pipelined_data, $(libpath)), BlasInt, (Ptr{Void}, Ptr{Void}), obj, data)
    end

    function SWP_perf_drop_with_communication(obj, chunk_size, num_chunks)
      ccall((:SWP_perf_drop_with_communication, $(libpath)), BlasInt, (Ptr{Void}, BlasInt, BlasInt), obj, chunk_size, num_chunks)
    end

    function insert_task(obj, flag)
      ccall((:insert_task, $(libpath)), BlasInt, (Ptr{Void},Cint), obj, flag)
    end 


# constructors for each LD functions
    function new_dgemm(order::BlasChar, transA::BlasChar, transB::BlasChar, 
        M::BlasInt, N::BlasInt, K::BlasInt, alpha::Float64, 
        Ap::Ptr{Float64}, lda::BlasInt, Bp::Ptr{Float64}, ldb::BlasInt, 
        beta::Float64, Cp::Ptr{Float64}, ldc::BlasInt)
      ccall((:new_dgemm_LD, $(libpath)), Ptr{Void}, 
        (Uint8, Uint8, Uint8, BlasInt, BlasInt, BlasInt, Float64, Ptr{Float64}, BlasInt, 
         Ptr{Float64}, BlasInt, Float64, Ptr{Float64}, BlasInt),
        order, transA, transB, M, N, K, alpha, Ap, lda, Bp, ldb, beta, Cp, ldc)
    end

    function new_dgetrf(matrix_layout::Int, m::BlasInt, n::BlasInt, a::Ptr{Float64}, 
        lda::BlasInt, ipiv::Ptr{BlasInt})
      ccall((:new_dgetrf_LD, $(libpath)), Ptr{Void}, 
          (Int, BlasInt, BlasInt, Ptr{Float64}, BlasInt, Ptr{BlasInt}),
          matrix_layout, m, n, a, lda, ipiv)
    end

    function new_dlaswp(matrix_order::Int, n::BlasInt, a::Ptr{Float64}, lda::BlasInt, 
        k1::BlasInt, k2::BlasInt, ipiv::Ptr{BlasInt}, incx::BlasInt)
      ccall((:new_dlaswp_LD, $(libpath)), Ptr{Void},
        (Int, BlasInt, Ptr{Float64}, BlasInt, BlasInt, BlasInt, Ptr{BlasInt}, BlasInt),
        matrix_order, n, a, lda, k1, k2, ipiv, incx);
    end

    function new_dtrsm(layout::BlasChar, side::BlasChar, uplo::BlasChar, transa::BlasChar, diag::BlasChar, 
        m::BlasInt, n::BlasInt, alpha::Float64, a::Ptr{Float64}, lda::BlasInt, b::Ptr{Float64}, ldb::BlasInt)
      ccall((:new_dtrsm_LD, $(libpath)), Ptr{Void}, 
        (BlasChar, BlasChar, BlasChar, BlasChar, BlasChar, BlasInt, BlasInt, Float64, Ptr{Float64}, 
         BlasInt, Ptr{Float64}, BlasInt),
        layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    end

    function new_dpotrf(matrix_layout::Int, uplo::BlasChar, n::BlasInt, a::Ptr{Float64}, lda::BlasInt)
      ccall((:new_dpotrf_LD, $(libpath)), Ptr{Void}, 
        (Int, BlasChar, BlasInt, Ptr{Float64}, BlasInt ),
        matrix_layout, uplo, n, a, lda);
    end

    function new_dsyrk(layout::BlasChar, uplo::BlasChar, trans::BlasChar, N::BlasInt, 
        K::BlasInt, alpha::Float64, Ap::Ptr{Float64}, lda::BlasInt, beta::Float64, 
        Cp::Ptr{Float64}, ldc::BlasInt)
      ccall((:new_dsyrk_LD, $(libpath)), Ptr{Void}, 
        (Uint8, Uint8, Uint8, BlasInt, BlasInt, Float64, Ptr{Float64}, BlasInt, 
        Float64, Ptr{Float64}, BlasInt), 
        layout, uplo, trans, N, K, alpha, Ap, lda, beta, Cp, ldc)
    end

    function new_dgeqr2( matrix_layout::Int, m::BlasInt, n::BlasInt,
        a::Ptr{Float64}, lda::BlasInt, tau::Ptr{Float64} )
      ccall((:new_dgeqr2_LD, $(libpath)), Ptr{Void}, 
        (Int, BlasInt, BlasInt, Ptr{Float64}, BlasInt, Ptr{Float64}),
        matrix_layout, m, n, a, lda, tau )
    end

    function new_dlarft( matrix_layout::Int, direct::BlasChar, storev::BlasChar,
        n::BlasInt, k::BlasInt,  v::Ptr{Float64},
        ldv::BlasInt, tau::Ptr{Float64}, t::Ptr{Float64},
        ldt::BlasInt )
      ccall((:new_dlarft_LD, $(libpath)), Ptr{Void}, 
        (Int,  Uint8, Uint8, BlasInt, BlasInt, Ptr{Float64}, BlasInt, Ptr{Float64}, Ptr{Float64},		BlasInt ),
        matrix_layout, direct, storev, n, k, v, ldv, tau, t, ldt )
    end

    function new_dlarfb( matrix_layout::Int, side::BlasChar, trans::BlasChar, direct::BlasChar, 
        storev::BlasChar, m::BlasInt, n::BlasInt, k::BlasInt,  v::Ptr{Float64},
        ldv::BlasInt, t::Ptr{Float64}, ldt::BlasInt, c::Ptr{Float64}, 
        ldc::BlasInt )
      ccall((:new_dlarfb_LD, $(libpath)), Ptr{Void}, 
        (Int, Uint8, Uint8, Uint8, Uint8,  BlasInt, BlasInt, BlasInt, Ptr{Float64}, 
        BlasInt, Ptr{Float64}, BlasInt, Ptr{Float64}, BlasInt ),
        matrix_layout, side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc )
    end
  end
end

end
