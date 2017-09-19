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
    if (fun === :abs) && (isFloat || isDouble || isComplex || isInt)
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
    if fun.mod==Base || fun.mod==ParallelAccelerator.API
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
    if fun.name==:throw || fun.name==:error
        s = "(throw(\"Julia throw() or error() called.\"), 0)"
    end
    return s
end

function pattern_match_call_throw(fun::Symbol, input, linfo)
    s = ""
    if fun==:throw || fun==:error
        s = "(throw(\"Julia throw() or error() called.\"), 0)"
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

function pattern_match_call_rand(linfo, fun, args...)
    @dprintln(3,"pattern_match_call_rand ", fun)
    res = ""
    if isBaseFunc(fun, :rand)
        if USE_OMP==1
            res = "cgen_distribution(cgen_rand_generator[omp_get_thread_num()]);\n"
        else
            res = "cgen_distribution(cgen_rand_generator);\n"
        end
    end
    @dprintln(3,"pattern_match_call_rand res = ", res)
    return res
end

function pattern_match_call_randn(linfo, fun, args...)
    @dprintln(3,"pattern_match_call_randn ", fun)
    res = ""
    if isBaseFunc(fun, :randn)
        if USE_OMP==1
            res = "cgen_n_distribution(cgen_rand_generator[omp_get_thread_num()]);\n"
        else
            res = "cgen_n_distribution(cgen_rand_generator);\n"
        end
    end
    @dprintln(3,"pattern_match_call_randn res = ", res)
    return res
end

function pattern_match_call_reshape(fun, inp::Any, shape::RHSVar, linfo)
    res = ""
    if isBaseFunc(fun, :reshape) || fun==GlobalRef(ParallelAccelerator.API,:reshape)
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

function pattern_match_call_gemm(fun::GlobalRef, C::RHSVar, tA::Char, tB::Char, A::RHSVar, B::RHSVar, alpha, beta, linfo)
    if fun.mod!=Base.LinAlg || fun.name!=:gemm_wrapper!
        return ""
    end
    calpha = from_expr(alpha,linfo)
    cbeta = from_expr(beta,linfo)
    cblas_fun = ""
    a_typ = getSymType(A, linfo)
    b_typ = getSymType(B, linfo)
    c_typ = getSymType(C, linfo)
    if !isArrayType(a_typ) || !isArrayType(b_typ) || !isArrayType(c_typ)
        return ""
    end
    typ = eltype(a_typ)
    if eltype(b_typ) != typ || eltype(c_typ) != typ
        return ""
    end
    if typ==Float32
        cblas_fun = "cblas_sgemm"
    elseif typ==Float64
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


    if ParallelAccelerator.getMklLib()!="" || ParallelAccelerator.getOpenblasLib()!="" || ParallelAccelerator.getSysBlas()==1
        s *= "$(cblas_fun)((CBLAS_ORDER)$(CblasColMajor),(CBLAS_TRANSPOSE)$(_tA),(CBLAS_TRANSPOSE)$(_tB),$m,$n,$k,$calpha,
        $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb, $cbeta, $(from_expr(C,linfo)).data, $ldc)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow.
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        s *= "cgen_$(cblas_fun)($(from_expr(tA!='N',linfo)), $(from_expr(tB!='N',linfo)), $m,$n,$k, $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb, $(from_expr(C,linfo)).data, $ldc)"
    end

    return s
end

function pattern_match_call_gemm(fun::ANY, C::ANY, tA::ANY, tB::ANY, A::ANY, B::ANY,alpha::ANY,beta::ANY,linfo)
    return ""
end

function pattern_match_call_gemv(fun::GlobalRef, y::RHSVar, tA::Char, A::RHSVar, x::RHSVar,linfo)
    if !((fun.mod==Base.LinAlg || fun.mod==Base.LinAlg.BLAS) && fun.name==:gemv!)
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


    if ParallelAccelerator.getMklLib()!="" || ParallelAccelerator.getOpenblasLib()!="" || ParallelAccelerator.getSysBlas()==1
        s *= "$(cblas_fun)((CBLAS_ORDER)$(CblasColMajor),(CBLAS_TRANSPOSE)$(_tA),$m,$n, 1.0,
        $(from_expr(A,linfo)).data, $lda, $(from_expr(x,linfo)).data, 1, 0.0, $(from_expr(y,linfo)).data, 1)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix-vector multiplication might be slow.
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        s *= "cgen_$(cblas_fun)($(from_expr(tA!='N',linfo)), $m,$n, $(from_expr(A,linfo)).data, $lda, $(from_expr(y,linfo)).data, $(from_expr(x,linfo)).data)"
    end

    return s
end

function pattern_match_call_gemv(fun::ANY, C::ANY, tA::ANY, A::ANY, B::ANY,linfo)
    return ""
end

function pattern_match_call_chol(fun::GlobalRef, A::RHSVar, linfo)
    if fun.mod!=Base.LinAlg || fun.name!=:chol
        return ""
    end

    cblas_fun = ""
    typ = eltype(getSymType(A, linfo))

    if typ==Float32
        lapack_fun = "LAPACKE_spotrf"
    elseif typ==Float64
        lapack_fun = "LAPACKE_dpotrf"
    else
        return ""
    end

    s = "$(from_expr(A,linfo)); "

    n = from_arraysize(A,1,linfo)


    lda = from_arraysize(A,1,linfo)


    LAPACK_COL_MAJOR = 102
    uplo = 'U' #vUL==Val{:U} ? 'U' : 'L'


    if ParallelAccelerator.getMklLib()!="" || ParallelAccelerator.getOpenblasLib()!="" || ParallelAccelerator.getSysBlas()==1
        s *= "$(lapack_fun)($(LAPACK_COL_MAJOR), '$uplo', $n, $(from_expr(A,linfo)).data, $lda)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow.
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        error("MKL LAPACK required for cholesky (TODO: support other lapack libraries and include a sequential implementation)")
        #s *= "cgen_$(cblas_fun)($(from_expr(tA!='N',linfo)), $m,$n, $(from_expr(A,linfo)).data, $lda, $(from_expr(y,linfo)).data, $(from_expr(x,linfo)).data)"
    end

    return s
end

function pattern_match_call_chol(fun::ANY, C::ANY, linfo)
    return ""
end

# # 0.4 legacy code not needed anymore
# function pattern_match_assignment_chol(lhs::LHSVar, rhs::Expr, linfo)
#     call = ""
#     if isCall(rhs) || isInvoke(rhs)
#         fun = getCallFunction(rhs)
#         args = getCallArguments(rhs)
#         if length(args) == 2
#             call *= pattern_match_call_chol(fun,args[1],args[2],linfo)
#         end
#     end
#     if call!=""
#         return from_expr(lhs,linfo)*call
#     end
#     return ""
# end
#
# function pattern_match_assignment_chol(lhs::ANY, rhs::ANY, linfo)
#     return ""
# end

function pattern_match_assignment_transpose(lhs::LHSVar, rhs::Expr, linfo)
    @dprintln(3, "pattern_match_assignment_transpose ", lhs, " ", rhs)
    call = ""
    if isCall(rhs) || isInvoke(rhs)
        fun = getCallFunction(rhs)
        args = getCallArguments(rhs)
        res = pattern_match_call_transpose(linfo, fun, lhs, args...)
        return res
    end
    return ""
end

function pattern_match_assignment_transpose(lhs::ANY, rhs::ANY, linfo)
    return ""
end

function pattern_match_call_trtrs(fun::GlobalRef, uplo::Char, trans::Char, diag::Char, A::RHSVar,B::RHSVar,  linfo)
    if fun.mod!=Base.LinAlg.LAPACK || fun.name!=:trtrs!
        return ""
    end

    cblas_fun = ""
    typ = eltype(getSymType(A, linfo))

    if typ==Float32
        lapack_fun = "LAPACKE_strtrs"
    elseif typ==Float64
        lapack_fun = "LAPACKE_dtrtrs"
    else
        return ""
    end

    s = "$(from_expr(B,linfo)); "

    n = from_arraysize(A,1,linfo)
    nrhs = from_arraysize(B,2,linfo)


    lda = from_arraysize(A,1,linfo)
    ldb = n

    LAPACK_COL_MAJOR = 102

    if ParallelAccelerator.getMklLib()!="" || ParallelAccelerator.getOpenblasLib()!="" || ParallelAccelerator.getSysBlas()==1
        s *= "$(lapack_fun)($(LAPACK_COL_MAJOR), '$uplo', '$trans', '$diag', $n, $nrhs, $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow.
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        error("MKL LAPACK required for triangular solution (TODO: support other lapack libraries and include a sequential implementation)")
        #s *= "cgen_$(cblas_fun)($(from_expr(tA!='N',linfo)), $m,$n, $(from_expr(A,linfo)).data, $lda, $(from_expr(y,linfo)).data, $(from_expr(x,linfo)).data)"
    end

    return s
end

function pattern_match_call_trtrs(fun::ANY, C::ANY, tA::ANY,A::ANY, t::ANY, d::ANY, linfo)
    return ""
end

function pattern_match_call_copy!(linfo, fun, A, i, B, j, n)
    if isBaseFunc(fun, :copy!)
        "j2c_array_copyto(" *
          from_expr(A, linfo) * ", " *
          from_expr(i, linfo) * ", " *
          from_expr(B, linfo) * ", " *
          from_expr(j, linfo) * ", " *
          from_expr(n, linfo) * ")"
    else
        ""
    end
end

function pattern_match_call_transpose(linfo, fun::GlobalRef, fun1::GlobalRef, B::RHSVar, A::RHSVar)
    pattern_match_call_transpose(linfo, fun, B, A)
end

function pattern_match_call_transpose(linfo, fun::GlobalRef, B::RHSVar, A::RHSVar)
    dprintln(3, "pattern_match_call_transpose, ", (fun, B, A), " ParallelAccelerator.getMklLib()=", ParallelAccelerator.getMklLib(), " openblas=",ParallelAccelerator.getOpenblasLib(), " ParallelAccelerator.getSysBlas()=",ParallelAccelerator.getSysBlas())
    if !(fun.mod==Base && fun.name==:transpose! || fun.name ==:transpose_f! || fun.name == :transpose)
        return ""
    end
    blas_fun = ""
    typ = eltype(getSymType(A, linfo))
    ctyp = toCtype(typ)

    if typ==Float32
        blas_fun = "somatcopy"
    elseif typ==Float64
        blas_fun = "domatcopy"
    else
        blas_fun = ""
    end

    #s = "$(from_expr(B,linfo)); "
    s = ""

    m = from_arraysize(A,1,linfo)
    n = from_arraysize(A,2,linfo)

    lda = from_arraysize(A,1,linfo)
    ldb = from_arraysize(A,2,linfo)

    if ParallelAccelerator.getMklLib()!="" && blas_fun!=""
        s *= "mkl_$(blas_fun)('C','T',$m,$n, 1.0,
             $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb)"
    elseif (ParallelAccelerator.getOpenblasLib()!="" || ParallelAccelerator.getSysBlas()==1) && blas_fun!=""
        s *= "cblas_$(blas_fun)(CblasColMajor,CblasTrans,$m,$n, 1.0,
             $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb)"
    else
        #println("""WARNING: MKL and OpenBLAS not found. Matrix-vector multiplication might be slow.
        #Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.""")
        #s *= "cgen_$(blas_fun)($m,$n,
        #     $(from_expr(A,linfo)).data, $lda, $(from_expr(B,linfo)).data, $ldb)"
        s *= "for(int i=0; i<$m; i++) {\n"
        s *= "    for(int j=0; j<$n; j++) {\n"
        s *= "     $(from_expr(B,linfo)).data[j+i*$ldb] = $(from_expr(A,linfo)).data[i+j*$lda];\n"
        s *= "    }\n"
        s *= "}\n"
    end
    s = (fun.name == :transpose ? (from_expr(B,linfo) * " = j2c_array<$ctyp>::new_j2c_array_2d(NULL, $n, $m);\n"): "") * s
    s = from_expr(B,linfo)*"; "*s
    return s
end

function pattern_match_call_transpose(args...)
    return ""
end

function pattern_match_call_linalgtypeof(fun::GlobalRef, C::ANY,linfo)
    if fun.mod==Base.LinAlg && fun.name==:typeof
        return " "
    end
    return ""
end

function pattern_match_call_linalgtypeof(fun::ANY, C::ANY,linfo)
    return ""
end

function pattern_match_call_vecnorm(fun::GlobalRef, y::RHSVar, p::Int,linfo)
    if fun.mod!=Base.LinAlg || fun.name!=:vecnorm
        return ""
    end
    cblas_fun = ""
    typ = eltype(getSymType(y, linfo))

    if typ==Float32
        cblas_fun = "cblas_s"
    elseif typ==Float64
        cblas_fun = "cblas_d"
    else
        return ""
    end

    if p==1
        cblas_fun *= "asum"
    elseif p==2
        cblas_fun *= "nrm2"
    else
        println("norm ",p)
        error("vector norm not support")
    end

    s = ""

    n = from_arraysize(y,1,linfo)


    if ParallelAccelerator.getMklLib()!="" || ParallelAccelerator.getOpenblasLib()!="" || ParallelAccelerator.getSysBlas()==1
        s *= "$(cblas_fun)( $n, $(from_expr(y,linfo)).data, 1)"
    else
        #println("WARNING: MKL and OpenBLAS not found. Matrix-vector multiplication might be slow.
        #Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        s *= "cgen_$(cblas_fun)( $n, $(from_expr(y,linfo)).data)"
    end

    return s
end

function pattern_match_call_vecnorm(fun::ANY, C::ANY, tA::ANY,linfo)
    return ""
end

function pattern_match_call_reduce_oprs(fun::GlobalRef, x, y,linfo)
    if fun.name in [:+, :*, :max, :min, :|, :&]
        cx = from_expr(x,linfo)
        cy = from_expr(y,linfo)
        if fun.name in [:min,:max]
            # std::min/max are picky about types
            tx = toCtype(getType(x,linfo))
            ty = toCtype(getType(y,linfo))
            return "std::$(fun.name)(($tx)$cx,($ty)$cy)"
        else
            return "($cx$(fun.name)$cy)"
        end
    end
    return ""
end

function pattern_match_call_reduce_oprs(fun::ANY, x::ANY, y::ANY,linfo)
    return ""
end


function pattern_match_call_subarray_lastdim(func::GlobalRef, arr::RHSVar, index::Union{Int,RHSVar}, linfo)
    if func==GlobalRef(ParallelAccelerator.API,:SubArrayLastDimRead) || func==GlobalRef(ParallelAccelerator.API,:SubArrayLastDimWrite)
        arr_typ = getType(arr, linfo)
        typ = eltype(arr_typ)
        ctyp = ParallelAccelerator.CGen.toCtype(typ)
        dims = ndims(arr_typ)
        carr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        cindex = ParallelAccelerator.CGen.from_expr(toLHSVarOrNum(index), linfo)
        shape = mapfoldl(i->from_arraysize(arr,i,linfo), (a,b)->a *","*b, 1:dims-1)
        low_dims_size = mapfoldl(i->from_arraysize(arr,i,linfo), (a,b)->a *"*"*b, 1:dims-1)
        pointer = "&($carr.data[$low_dims_size*($cindex-1)])"
        return "j2c_array<$ctyp>::new_j2c_array_$(dims-1)d($pointer, $shape)"
    end
    return ""
end

pattern_match_call_subarray_lastdim(func::ANY, arr, index, linfo) = ""

function pattern_match_call_set_zeros(func::GlobalRef, arr::RHSVar, size, linfo)
    if func==GlobalRef(ParallelAccelerator.API,:set_zeros)
        carr = from_expr(toLHSVar(arr), linfo)
        ctyp = toCtype(eltype(getSymType(arr,linfo)))
        csize = from_expr(size, linfo)
        return "memset($carr.data, 0, sizeof($ctyp)*$csize)"
    end
    return ""
end

pattern_match_call_set_zeros(func::ANY, arr::ANY, size, linfo) = ""

function pattern_match_call(ast::Array{Any, 1},linfo)
    @dprintln(3,"pattern matching ",ast)
    s = ""

    if(length(ast)==2)
        s = pattern_match_call_throw(ast[1],ast[2],linfo)
        s *= pattern_match_call_math(ast[1],ast[2],linfo)
        s *= pattern_match_call_linalgtypeof(ast[1],ast[2],linfo)
        s *= pattern_match_call_chol(ast[1],ast[2],linfo)
    end

    if s=="" && (length(ast)==3) # randn! call has 3 args
        #sa*= pattern_match_call_powersq(ast[1],ast[2], ast[3])
        s *= pattern_match_call_set_zeros(ast[1],ast[2],ast[3],linfo)
        s *= pattern_match_call_reshape(ast[1],ast[2],ast[3],linfo)
        s *= pattern_match_call_transpose(linfo, ast...)
        s *= pattern_match_call_vecnorm(ast[1],ast[2],ast[3],linfo)
        s *= pattern_match_call_reduce_oprs(ast[1],ast[2],ast[3],linfo)
        s *= pattern_match_call_subarray_lastdim(ast[1],ast[2],ast[3], linfo)
    end
    if s=="" && (length(ast)>=1) # rand can have 1 or more arg
        s *= pattern_match_call_transpose(linfo, ast...)
        s *= pattern_match_call_randn(linfo, ast...)
        s *= pattern_match_call_rand(linfo, ast...)
    end
    # gemv calls have 5 args
    if s=="" && (length(ast)==5)
        s *= pattern_match_call_gemv(ast[1],ast[2],ast[3],ast[4],ast[5],linfo)
    end
    # gemm calls have 6 args
    if s=="" && (length(ast)==6)
        s *= pattern_match_call_copy!(linfo, ast...)
        s *= pattern_match_call_gemm(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6], 1.0, 0.0, linfo)
        s *= pattern_match_call_trtrs(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],linfo)
    end
    if s=="" && (length(ast)==8)
        s *= pattern_match_call_gemm(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6], ast[7], ast[8], linfo)
    end
    return s
end


function from_assignment_match_hvcat(lhs, rhs::Expr, linfo)
    s = ""
    # if this is a hvcat call, the array should be allocated and initialized
    if (isCall(rhs) || isInvoke(rhs)) && (isBaseFunc(getCallFunction(rhs),:typed_hvcat) || checkGlobalRefName(getCallFunction(rhs),:hvcat))
        @dprintln(3,"Found hvcat assignment: ", lhs," ", rhs)

        is_typed::Bool = isBaseFunc(getCallFunction(rhs),:typed_hvcat)

        rows = Int64[]
        values = Any[]
        typ = "double"
        args = getCallArguments(rhs)

        if is_typed
            atyp = args[1]
            if isa(atyp, GlobalRef)
                atyp = eval(args[1].name)
            end
            @assert isa(atyp, DataType) ("hvcat expects the first argument to be a type, but got " * args[1])
            typ = toCtype(atyp)
            tuple_arg = args[2]
            if isa(tuple_arg, Tuple)
                rows = tuple_arg
            else
                rows = lstate.tupleTable[tuple_arg]
            end
            values = args[3:end]
        else
            tuple_arg = args[1]
            if isa(tuple_arg, Tuple)
                rows = tuple_arg
            else
                rows = lstate.tupleTable[tuple_arg]
            end
            values = args[2:end]
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
    if (isCall(rhs) || isInvoke(rhs)) && isBaseFunc(getCallFunction(rhs), :cat_t)
        args = getCallArguments(rhs)
        dims = args[1]
        @assert dims==2 "CGen: only 2d cat_t() is supported now"
        size = length(args[3:end])
        typ = toCtype(eval(args[2].name))
        s *= from_expr(lhs,linfo) * " = j2c_array<$typ>::new_j2c_array_$(dims)d(NULL, 1,$size);\n"
        values = args[3:end]
        s *= mapfoldl((i) -> from_setindex([lhs,values[i],i],linfo)*";", (a, b) -> "$a $b", 1:length(values))
    end
    return s
end

function from_assignment_match_cat_t(lhs, rhs::ANY, linfo)
    return ""
end

function from_assignment_match_hcat(lhs, rhs::Expr, linfo)
    s = ""
    if (isCall(rhs) || isInvoke(rhs)) && isBaseFunc(getCallFunction(rhs), :hcat)
        args = getCallArguments(rhs)
        in_typ = getType(args[1], linfo)
        # hcat of single values like hcat(1,2,3) => 2D array [1 2 3]
        # used in quant example
        if !(in_typ<:Array)
            return single_value_hcat(lhs, args, linfo)
        end
        for a in args
            atyp = getType(a, linfo)
            @assert atyp<:Array && ndims(atyp)==1 "CGen only supports hcat of 1D arrays"
        end
        typ = eltype(getType(lhs, linfo))
        size = length(args)
        ctyp = toCtype(typ)
        len = from_arraysize(args[1],1,linfo)
        clhs = from_expr(lhs,linfo)
        s *= "$clhs = j2c_array<$ctyp>::new_j2c_array_2d(NULL, $len, $size);\n"
        for j in 1:size
            s *= "for(int i=0; i<$len; i++) {\n"
            arr = from_expr(args[j],linfo)
            s *= "$clhs.data[$(j-1)*$len+i] = $arr.data[i];\n"
            s *= "}\n"
        end
    end
    return s
end

function from_assignment_match_hcat(lhs, rhs::ANY, linfo)
    return ""
end

function single_value_hcat(lhs, args, linfo)
    for a in args
        atyp = getType(a, linfo)
        @assert !(atyp<:Array) "CGen invalid hcat input (single_value_hcat)"
    end
    typ = getType(args[1], linfo)
    size = length(args)
    ctyp = toCtype(typ)
    clhs = from_expr(lhs,linfo)
    s = "$clhs = j2c_array<$ctyp>::new_j2c_array_2d(NULL, 1, $size);\n"
    for j in 1:size
        var = from_expr(args[j],linfo)
        s *= "$clhs.data[$(j-1)] = $var;\n"
    end
    return s
end


function from_assignment_match_vcat(lhs, rhs::Expr, linfo)
    s = ""
    if (isCall(rhs) || isInvoke(rhs)) && isBaseFunc(getCallFunction(rhs), :vcat)
        args = getCallArguments(rhs)
        for a in args
            atyp = getType(a, linfo)
            @assert atyp<:Array && (ndims(atyp)==1 || ndims(atyp)==2) "CGen only supports vcat of 1D and 2D arrays"
        end
        num_dims = ndims(getType(args[1], linfo))
        typ = eltype(getType(args[1], linfo))
        ctyp = toCtype(typ)
        clhs = from_expr(lhs,linfo)
        # get total size of array: size(a1)+size(a2)+...
        csize = "("* mapfoldl(a->from_arraysize(a,1,linfo),(a,b)->"$a+$b",args) *")"
        c_num_cols = "1"
        if num_dims==2
            c_num_cols = from_arraysize(args[1],2,linfo)
            csize *= ", $c_num_cols"
        end
        s *= "{\n"
        s *= "$clhs = j2c_array<$ctyp>::new_j2c_array_$(num_dims)d(NULL, $csize);\n"
        s *= "int64_t __cgen_curr_ind = 0;\n"
        s *= "for(int64_t j=0; j<$c_num_cols; j++){\n"
        for arr in args
            carr = from_expr(arr, linfo)
            col_size = from_arraysize(arr,1,linfo)
            s *= "for(int64_t i=0; i<$col_size; i++){\n"
            s *= "  $clhs.data[__cgen_curr_ind++] = $carr.data[j*$col_size+i];\n"
            s *= "}\n"
        end
        s *= "}\n"
        s *= "}\n"
    end
    return s
end

function from_assignment_match_vcat(lhs, rhs::ANY, linfo)
    return ""
end

function from_assignment_match_iostream(lhs, rhs::GlobalRef, linfo)
    s = ""
    ltype = getType(lhs, linfo)
    @dprintln(3, "from_assignment_match_iostream ltype = ", ltype)
    if (ltype == IOStream)
        if rhs.mod == Base && rhs.name == :STDOUT
            lhsO = from_expr(lhs, linfo)
            s *= lhsO * ".handle = (void**)&(std::cout);"
        end
    end
    return s
end

function from_assignment_match_iostream(lhs, rhs::ANY, linfo)
    return ""
end
