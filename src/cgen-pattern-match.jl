#=
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
=# 


# math functions
libm_math_functions = Set([:sin, :cos, :tan, :asin, :acos, :acosh, :atanh, :log, :log2, :log10, :lgamma, :log1p,:asinh,:atan,:cbrt,:cosh,:erf,:exp,:expm1,:sinh,:sqrt,:tanh, :isnan])
#using Debug

function pattern_match_call_math(fun::Symbol, input::AbstractString, typ::Type, linfo)
    s = ""
    isDouble = typ == Float64 
    isFloat = typ == Float32
    isComplex = typ <: Complex
    isInt = typ <: Integer
    if in(fun,libm_math_functions) && (isFloat || isDouble || isComplex)
        @dprintln(3,"FOUND ", fun)
        s = string(fun)*"("*input*");"
    end

    # abs() needs special handling since fabs() in math.h should be called for floats
    if is(fun,:abs) && (isFloat || isDouble || isComplex || isInt)
      @dprintln(3,"FOUND ", fun)
      fname = (isInt || isComplex) ? "abs" : (isFloat ? "fabsf" : "fabs")
      s = fname*"("*input*");"
    end
    return s
end

function pattern_match_call_math(fun::Symbol, input::RHSVar, linfo)
  pattern_match_call_math(fun, from_expr(input, linfo), getType(input, linfo), linfo)
end


function pattern_match_call_math(fun::GlobalRef, input, linfo)
    fun = Base.resolve(fun)
    if fun.mod == Base 
        pattern_match_call_math(fun.name, input,linfo)
    else
        return ""
    end
end

function pattern_match_call_math(fun::ANY, input::ANY, linfo)
    return ""
end

function pattern_match_call_throw(fun::GlobalRef, input, linfo)
    s = ""
    if fun.name==:throw
        s = "throw(\"Julia throw() called.\")"
    end
    return s
end

function pattern_match_call_throw(fun::ANY, input::ANY, linfo)
    return ""
end

function pattern_match_call_powersq(fun, x::Number, y::Integer, linfo)
    s = ""
    if isBaseFunc(fun, :power_by_squaring)
        s = "cgen_pown("*from_expr(x,linfo)*","*from_expr(y,linfo)*")"
    end
    return s
end

function pattern_match_call_powersq(fun::ANY, x::ANY, y::ANY,linfo)
    return ""
end

function pattern_match_call_rand(linfo, fun, RNG::Any, args...)
    res = ""
    if isBaseFunc(fun, :rand!)
        if USE_OMP==1
            res = "cgen_distribution(cgen_rand_generator[omp_get_thread_num()]);\n"
        else
            res = "cgen_distribution(cgen_rand_generator);\n"
        end
    end
    return res 
end

function pattern_match_call_randn(fun, RNG::Any, IN::Any,linfo)
    res = ""
    if isBaseFunc(fun, :randn!)
        if USE_OMP==1
            res = "cgen_n_distribution(cgen_rand_generator[omp_get_thread_num()]);\n"
        else
            res = "cgen_n_distribution(cgen_rand_generator);\n"
        end
    end
    return res 
end

function pattern_match_call_reshape(fun, inp::Any, shape::RHSVar, linfo)
    res = ""
    if isBaseFunc(fun, :reshape)
        typ = getSymType(shape, linfo)
        if istupletyp(typ)
            dim = length(typ.parameters)
            sh = from_expr(shape,linfo)
            shapes = mapfoldl(i->sh*".f"*string(i-1), (a,b) -> a*","*b, 1:dim)
            res = from_expr(inp,linfo) * ".reshape(" * shapes * ");\n"
        else
            error("call to reshape expects a tuple, but got ", typ)
        end
    end
    return res 
end

function pattern_match_call_reshape(fun::ANY, inp::ANY, shape::ANY,linfo)
    return ""
end

function getSymType(a, linfo)
    return lstate.symboltable[lookupVariableName(a, linfo)]
end

function pattern_match_call_gemm(fun::GlobalRef, C::RHSVar, tA::Char, tB::Char, A::RHSVar, B::RHSVar,linfo)
    if fun.mod!=Base.LinAlg || fun.name!=:gemm_wrapper!
        return ""
    end
    cblas_fun = ""
    typ = getSymType(A, linfo)
    if getSymType(B, linfo) != typ || getSymType(C, linfo) != typ
        return ""
    end
    if typ==Array{Float32,2}
        cblas_fun = "cblas_sgemm"
    elseif typ==Array{Float64,2}
        cblas_fun = "cblas_dgemm"
    else
        return ""
    end
    s = "$(from_expr(C,linfo)); "
    # GEMM wants dimensions after possible transpose
    m = (tA == 'N') ? from_arraysize(A,1,linfo) : from_arraysize(A,2,linfo) 
    k = (tA == 'N') ? from_arraysize(A,2,linfo) : from_arraysize(A,1,linfo) 
    n = (tB == 'N') ? from_arraysize(B,2,linfo) : from_arraysize(B,1,linfo)

    lda = from_arraysize(A,1,linfo)
    ldb = from_arraysize(B,1,linfo)
    ldc = m

    CblasNoTrans = 111 
    CblasTrans = 112 
    _tA = tA == 'N' ? CblasNoTrans : CblasTrans
    _tB = tB == 'N' ? CblasNoTrans : CblasTrans
    CblasColMajor = 102


    if mkl_lib!="" || openblas_lib!="" || sys_blas==1
        s *= "$(cblas_fun)((CBLAS_ORDER)$(CblasColMajor),(CBLAS_TRANSPOSE)$(_tA),(CBLAS_TRANSPOSE)$(_tB),$m,$n,$k,1.0,
        $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb, 0.0, $(from_expr(C,linfo)).data, $ldc)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow. 
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        s *= "cgen_$(cblas_fun)($(from_expr(tA!='N',linfo)), $(from_expr(tB!='N',linfo)), $m,$n,$k, $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb, $(from_expr(C,linfo)).data, $ldc)"
    end

    return s
end

function pattern_match_call_gemm(fun::ANY, C::ANY, tA::ANY, tB::ANY, A::ANY, B::ANY,linfo)
    return ""
end

function pattern_match_call_gemv(fun::GlobalRef, y::RHSVar, tA::Char, A::RHSVar, x::RHSVar,linfo)
    if fun.mod!=Base.LinAlg || fun.name!=:gemv!
        return ""
    end
    cblas_fun = ""
    typ = eltype(getSymType(A, linfo))
    
    if typ==Float32
        cblas_fun = "cblas_sgemv"
    elseif typ==Float64
        cblas_fun = "cblas_dgemv"
    else
        return ""
    end
    
    s = "$(from_expr(y,linfo)); "

    m = from_arraysize(A,1,linfo) 
    n = from_arraysize(A,2,linfo)


    lda = from_arraysize(A,1,linfo)
    
    CblasNoTrans = 111 
    CblasTrans = 112 
    _tA = tA == 'N' ? CblasNoTrans : CblasTrans
    CblasColMajor = 102


    if mkl_lib!="" || openblas_lib!="" || sys_blas==1
        s *= "$(cblas_fun)((CBLAS_ORDER)$(CblasColMajor),(CBLAS_TRANSPOSE)$(_tA),$m,$n, 1.0,
        $(from_expr(A,linfo)).data, $lda, $(from_expr(x,linfo)).data, 1, 0.0, $(from_expr(y,linfo)).data, 1)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow. 
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        s *= "cgen_$(cblas_fun)($(from_expr(tA!='N',linfo)), $m,$n, $(from_expr(A,linfo)).data, $lda, $(from_expr(y,linfo)).data, $(from_expr(x,linfo)).data)"
    end

    return s
end

function pattern_match_call_gemv(fun::ANY, C::ANY, tA::ANY, A::ANY, B::ANY,linfo)
    return ""
end

function pattern_match_call(ast::Array{Any, 1},linfo)
    @dprintln(3,"pattern matching ",ast)
    s = ""

    if(length(ast)==2)
        s = pattern_match_call_throw(ast[1],ast[2],linfo)
        s *= pattern_match_call_math(ast[1],ast[2],linfo)
    end
    
    if(length(ast)==3) # randn! call has 3 args
        s *= pattern_match_call_randn(ast[1],ast[2],ast[3],linfo)
        #sa*= pattern_match_call_powersq(ast[1],ast[2], ast[3])
        s *= pattern_match_call_reshape(ast[1],ast[2],ast[3],linfo)
    end
    if(length(ast)>=2) # rand! has 2 or more args
        s *= pattern_match_call_rand(linfo, ast...)
    end
    # gemv calls have 5 args
    if(length(ast)==5)
        s *= pattern_match_call_gemv(ast[1],ast[2],ast[3],ast[4],ast[5],linfo)
    end
    # gemm calls have 6 args
    if(length(ast)==6)
        s = pattern_match_call_gemm(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],linfo)
    end
    return s
end


function from_assignment_match_hvcat(lhs, rhs::Expr, linfo)
    s = ""
    # if this is a hvcat call, the array should be allocated and initialized
    if rhs.head==:call && (isBaseFunc(rhs.args[1],:typed_hvcat) || checkGlobalRefName(rhs.args[1],:hvcat))
        @dprintln(3,"Found hvcat assignment: ", lhs," ", rhs)

        is_typed::Bool = isBaseFunc(rhs.args[1],:typed_hvcat)
        
        rows = Int64[]
        values = Any[]
        typ = "double"

        if is_typed
            atyp = rhs.args[2]
            if isa(atyp, GlobalRef) 
                atyp = eval(rhs.args[2].name)
            end
            @assert isa(atyp, DataType) ("hvcat expects the first argument to be a type, but got " * rhs.args[2])
            typ = toCtype(atyp)
            rows = lstate.tupleTable[rhs.args[3]]
            values = rhs.args[4:end]
        else

            rows = lstate.tupleTable[rhs.args[2]]
            values = rhs.args[3:end]
            atyp, arr_dims = parseArrayType(getSymType(lhs, linfo))
            typ = toCtype(atyp)
        end

        nr = length(rows)
        nc = rows[1] # all rows should have the same size
        s *= from_expr(lhs,linfo) * " = j2c_array<$typ>::new_j2c_array_2d(NULL, $nr, $nc);\n"
        s *= mapfoldl((i) -> from_setindex([lhs,values[i],convert(Int64,ceil(i/nc)),(i-1)%nc+1],linfo)*";", (a, b) -> "$a $b", 1:length(values))
    end
    return s
end

function from_assignment_match_hvcat(lhs, rhs::ANY, linfo)
    return ""
end

function from_assignment_match_cat_t(lhs, rhs::Expr, linfo)
    s = ""
    if rhs.head==:call && isa(rhs.args[1],GlobalRef) && rhs.args[1].name==:cat_t
        dims = rhs.args[2]
        @assert dims==2 "CGen: only 2d cat_t() is supported now"
        size = length(rhs.args[4:end])
        typ = toCtype(eval(rhs.args[3].name))
        s *= from_expr(lhs,linfo) * " = j2c_array<$typ>::new_j2c_array_$(dims)d(NULL, 1,$size);\n"
        values = rhs.args[4:end]
        s *= mapfoldl((i) -> from_setindex([lhs,values[i],i],linfo)*";", (a, b) -> "$a $b", 1:length(values))
    end
    return s
end

function from_assignment_match_cat_t(lhs, rhs::ANY, linfo)
    return ""
end


