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


function pattern_match_call_math(fun::TopNode, input::ASCIIString, typ::Type)
    s = ""
    isDouble = typ == Float64 
    isFloat = typ == Float32
    isComplex = typ <: Complex
    isInt = typ <: Integer
    if in(fun.name,libm_math_functions) && (isFloat || isDouble || isComplex)
        dprintln(3,"FOUND ", fun.name)
        s = string(fun.name)*"("*input*");"
    end

    # abs() needs special handling since fabs() in math.h should be called for floats
    if is(fun.name,:abs) && (isFloat || isDouble || isComplex || isInt)
      dprintln(3,"FOUND ", fun.name)
      fname = (isInt || isComplex) ? "abs" : (isFloat ? "fabsf" : "fabs")
      s = fname*"("*input*");"
    end
    return s
end

function pattern_match_call_math(fun::TopNode, input::GenSym)
  pattern_match_call_math(fun, from_expr(input), lstate.symboltable[input])
end


function pattern_match_call_math(fun::TopNode, input::SymbolNode)
  pattern_match_call_math(fun, from_expr(input), input.typ)
end

function pattern_match_call_math(fun::GlobalRef, input)
    return pattern_match_call_math(TopNode(fun.name), input)
end

function pattern_match_call_math(fun::ANY, input::ANY)
    return ""
end

function pattern_match_call_throw(fun::GlobalRef, input)
    s = ""
    if fun.name==:throw
        s = "throw(\"Julia throw() called.\")"
    end
    return s
end

function pattern_match_call_throw(fun::ANY, input::ANY)
    return ""
end

function pattern_match_call_powersq(fun::GlobalRef, x::Number, y::Integer)
    s = ""
    if fun.name==:power_by_squaring
        s = "cgen_pown("*from_expr(x)*","*from_expr(y)*")"
    end
    return s
end

function pattern_match_call_powersq(fun::ANY, x::ANY, y::ANY)
    return ""
end

function pattern_match_call_rand(fun::TopNode, RNG::Any, args...)
    res = ""
    if(fun.name==:rand!)
        res = "cgen_distribution(cgen_rand_generator);\n"
    end
    return res 
end

function pattern_match_call_rand(fun::ANY, RNG::ANY, args...)
    return ""
end

function pattern_match_call_randn(fun::TopNode, RNG::Any, IN::Any)
    res = ""
    if(fun.name==:randn!)
        res = "cgen_n_distribution(cgen_rand_generator);\n"
    end
    return res 
end

function pattern_match_call_randn(fun::ANY, RNG::ANY, IN::ANY)
    return ""
end

function pattern_match_call_reshape(fun::GlobalRef, inp::Any, shape::Union{SymbolNode,Symbol,GenSym})
    res = ""
    if(fun.mod == Base && fun.name==:reshape)
        typ = getSymType(shape)
        if istupletyp(typ)
            dim = length(typ.parameters)
            sh = from_expr(shape)
            shapes = mapfoldl(i->sh*".f"*string(i-1), (a,b) -> a*","*b, 1:dim)
            res = from_expr(inp) * ".reshape(" * shapes * ");\n"
        else
            error("call to reshape expects a tuple, but got ", typ)
        end
    end
    return res 
end

function pattern_match_call_reshape(fun::ANY, inp::ANY, shape::ANY)
    return ""
end

function getSymType(a::Union{Symbol,GenSym})
    return lstate.symboltable[a]
end

function getSymType(a::SymbolNode)
    return lstate.symboltable[a.name]
end

function pattern_match_call_gemm(fun::GlobalRef, C::SymAllGen, tA::Char, tB::Char, A::SymAllGen, B::SymAllGen)
    if fun.mod!=Base.LinAlg || fun.name!=:gemm_wrapper!
        return ""
    end
    cblas_fun = ""
    typ = getSymType(A)
    if getSymType(B)!=typ || getSymType(C)!=typ
        return ""
    end
    if typ==Array{Float32,2}
        cblas_fun = "cblas_sgemm"
    elseif typ==Array{Float64,2}
        cblas_fun = "cblas_dgemm"
    else
        return ""
    end
    s = "$(from_expr(C)); "
    m = (tA == 'N') ? from_arraysize(A,1) : from_arraysize(A,2) 
    k = (tB == 'N') ? from_arraysize(A,2) : from_arraysize(A,1) 
    n = (tB == 'N') ? from_arraysize(B,2) : from_arraysize(B,1)

    lda = from_arraysize(A,1)
    ldb = from_arraysize(B,1)
    ldc = m

    CblasNoTrans = 111 
    CblasTrans = 112 
    _tA = tA == 'N' ? CblasNoTrans : CblasTrans
    _tB = tB == 'N' ? CblasNoTrans : CblasTrans
    CblasColMajor = 102


    if mkl_lib!="" || openblas_lib!=""
        s *= "$(cblas_fun)((CBLAS_LAYOUT)$(CblasColMajor),(CBLAS_TRANSPOSE)$(_tA),(CBLAS_TRANSPOSE)$(_tB),$m,$n,$k,1.0,
        $(from_expr(A)).data, $lda, $(from_expr(B)).data, $ldb, 0.0, $(from_expr(C)).data, $ldc)"
    else
        println("WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow. 
        Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.")
        s *= "cgen_$(cblas_fun)($(from_expr(tA!='N')), $(from_expr(tB!='N')), $m,$n,$k, $(from_expr(A)).data, $lda, $(from_expr(B)).data, $ldb, $(from_expr(C)).data, $ldc)"
    end

    return s
end

function pattern_match_call_gemm(fun::ANY, C::ANY, tA::ANY, tB::ANY, A::ANY, B::ANY)
    return ""
end

function pattern_match_call_dist_init(f::TopNode)
    if f.name==:hps_dist_init
        return ";"#"MPI_Init(0,0);"
    else
        return ""
    end
end

function pattern_match_call_dist_init(f::Any)
    return ""
end

function pattern_match_reduce_sum(reductionFunc::DelayedFunc)
    if reductionFunc.args[1][1].args[2].args[1]==TopNode(:add_float)
        return true
    end
    return false
end

function pattern_match_call_dist_reduce(f::TopNode, var::SymbolNode, reductionFunc::DelayedFunc, output::Symbol)
    if f.name==:hps_dist_reduce
        mpi_type = ""
        if var.typ==Float64
            mpi_type = "MPI_DOUBLE"
        elseif var.typ==Float32
            mpi_type = "MPI_FLOAT"
        elseif var.typ==Int32
            mpi_type = "MPI_INT"
        elseif var.typ==Int64
            mpi_type = "MPI_LONG_LONG_INT"
        else
            throw("CGen unsupported MPI reduction type")
        end

        mpi_func = ""
        if pattern_match_reduce_sum(reductionFunc)
            mpi_func = "MPI_SUM"
        else
            throw("CGen unsupported MPI reduction function")
        end
                
        s="MPI_Reduce(&$(var.name), &$output, 1, $mpi_type, $mpi_func, 0, MPI_COMM_WORLD);"
        return s
    else
        return ""
    end
end

function pattern_match_call_dist_reduce(f::Any, v::Any, rf::Any, o::Any)
    return ""
end

"""
Generate code for HDF5 file open
"""
function pattern_match_call_data_src_open(f::Symbol, id::GenSym, data_var::Union{SymAllGen,AbstractString}, file_name::Union{SymAllGen,AbstractString}, arr::Symbol)
    s = ""
    if f==:__hps_data_source_HDF5_open
        num::AbstractString = from_expr(id.id)
    
        s = "hid_t plist_id_$num = H5Pcreate(H5P_FILE_ACCESS);\n"
        s *= "assert(plist_id_$num != -1);\n"
        s *= "herr_t ret_$num;\n"
        s *= "hid_t file_id_$num;\n"
        s *= "ret_$num = H5Pset_fapl_mpio(plist_id_$num, MPI_COMM_WORLD, MPI_INFO_NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "file_id_$num = H5Fopen("*from_expr(file_name)*", H5F_ACC_RDONLY, plist_id_$num);\n"
        s *= "assert(file_id_$num != -1);\n"
        s *= "ret_$num = H5Pclose(plist_id_$num);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t dataset_id_$num;\n"
        s *= "dataset_id_$num = H5Dopen2(file_id_$num, "*from_expr(data_var)*", H5P_DEFAULT);\n"
        s *= "assert(dataset_id_$num != -1);\n"
    end
    return s
end

function pattern_match_call_data_src_open(f::Any, v::Any, rf::Any, o::Any, arr::Any)
    return ""
end

"""
Generate code for text file open (no variable name input)
"""
function pattern_match_call_data_src_open(f::Symbol, id::GenSym, file_name::Union{SymAllGen,AbstractString}, arr::Symbol)
    s = ""
    if f==:__hps_data_source_TXT_open
        num::AbstractString = from_expr(id.id)
        file_name_str::AbstractString = from_expr(file_name)
        s = """
            MPI_File dsrc_txt_file_$num;
            int ierr_$num = MPI_File_open(MPI_COMM_WORLD, $file_name_str, MPI_MODE_RDONLY, MPI_INFO_NULL, &dsrc_txt_file_$num);
            assert(ierr_$num==0);
            """
    end
    return s
end

function pattern_match_call_data_src_open(f::Any, rf::Any, o::Any, arr::Any)
    return ""
end



function pattern_match_call_data_src_read(f::Symbol, id::GenSym, arr::Symbol, start::Symbol, count::Symbol)
    s = ""
    num::AbstractString = from_expr(id.id)
    
    if f==:__hps_data_source_HDF5_read    
        # assuming 1st dimension is partitined
        s =  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s *= "CGen_HDF5_start_$num[0] = $start;\n"
        s *= "CGen_HDF5_count_$num[0] = $count;\n"
        s *= "for(int i_CGen_dim=1; i_CGen_dim<data_ndim_$num; i_CGen_dim++) {\n"
        s *= "CGen_HDF5_start_$num[i_CGen_dim] = 0;\n"
        s *= "CGen_HDF5_count_$num[i_CGen_dim] = space_dims_$num[i_CGen_dim];\n"
        s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= "ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_SET, CGen_HDF5_start_$num, NULL, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t mem_dataspace_$num = H5Screate_simple (data_ndim_$num, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert (mem_dataspace_$num != -1);\n"
        s *= "hid_t xfer_plist_$num = H5Pcreate (H5P_DATASET_XFER);\n"
        s *= "assert(xfer_plist_$num != -1);\n"
        s *= "ret_$num = H5Dread(dataset_id_$num, H5T_NATIVE_DOUBLE, mem_dataspace_$num, space_id_$num, xfer_plist_$num, $arr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
    elseif f==:__hps_data_source_TXT_read
        # assuming 1st dimension is partitined
        s = """
            int64_t CGen_txt_start_$num = $start;
            int64_t CGen_txt_count_$num = $count;
            int64_t CGen_txt_end_$num = $start+$count;
            
            
            // std::cout<<"rank: "<<__hps_node_id<<" start: "<<CGen_txt_start_$num<<" end: "<<CGen_txt_end_$num<<" columnSize: "<<CGen_txt_col_size_$num<<std::endl;
            // if data needs to be sent left
            // still call MPI_Send if first character is new line
            int64_t CGen_txt_left_send_size_$num = 0;
            int64_t CGen_txt_tmp_curr_start_$num = CGen_txt_curr_start_$num;
            if(CGen_txt_start_$num>CGen_txt_curr_start_$num)
            {
                while(CGen_txt_tmp_curr_start_$num!=CGen_txt_start_$num)
                {
                    while(CGen_txt_buffer_$num[CGen_txt_left_send_size_$num]!=\'\\n\') 
                        CGen_txt_left_send_size_$num++;
                    CGen_txt_left_send_size_$num++; // account for \n
                    CGen_txt_tmp_curr_start_$num++;
                }
            }
            MPI_Request CGen_txt_MPI_request1_$num, CGen_txt_MPI_request2_$num;
            MPI_Status CGen_txt_MPI_status_$num;
            // send left
            if(__hps_node_id!=0)
            {
                MPI_Isend(&CGen_txt_left_send_size_$num, 1, MPI_LONG_LONG_INT, __hps_node_id-1, 0, MPI_COMM_WORLD, &CGen_txt_MPI_request1_$num);
                MPI_Isend(CGen_txt_buffer_$num, CGen_txt_left_send_size_$num, MPI_CHAR, __hps_node_id-1, 1, MPI_COMM_WORLD, &CGen_txt_MPI_request2_$num);
                // std::cout<<"rank: "<<__hps_node_id<<" sent left "<<CGen_txt_left_send_size_$num<<std::endl;
            }
            
            char* CGen_txt_right_buff_$num = NULL;
            int64_t CGen_txt_right_recv_size_$num = 0;
            // receive from right
            if(__hps_node_id!=__hps_num_pes-1)
            {
                MPI_Recv(&CGen_txt_right_recv_size_$num, 1, MPI_LONG_LONG_INT, __hps_node_id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CGen_txt_right_buff_$num = new char[CGen_txt_right_recv_size_$num];
                MPI_Recv(CGen_txt_right_buff_$num, CGen_txt_right_recv_size_$num, MPI_CHAR, __hps_node_id+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // std::cout<<"rank: "<<__hps_node_id<<" received right "<<CGen_txt_right_recv_size_$num<<std::endl;
            }
            
            if(__hps_node_id!=0)
            {
                MPI_Wait(&CGen_txt_MPI_request1_$num, &CGen_txt_MPI_status_$num);
                MPI_Wait(&CGen_txt_MPI_request2_$num, &CGen_txt_MPI_status_$num);
            }
            
            // if data needs to be sent right
            // still call MPI_Send if first character is new line
            int64_t CGen_txt_right_send_size_$num = 0;
            int64_t CGen_txt_tmp_curr_end_$num = CGen_txt_curr_end_$num;
            if(__hps_node_id!=__hps_num_pes-1 && CGen_txt_curr_end_$num>=CGen_txt_end_$num)
            {
                while(CGen_txt_tmp_curr_end_$num!=CGen_txt_end_$num-1)
                {
                    // -1 to account for \0
                    while(CGen_txt_buffer_$num[CGen_txt_buff_size_$num-CGen_txt_right_send_size_$num-1]!=\'\\n\') 
                        CGen_txt_right_send_size_$num++;
                    CGen_txt_tmp_curr_end_$num--;
                    // corner case, last line doesn't have \'\\n\'
                    if (CGen_txt_tmp_curr_end_$num!=CGen_txt_end_$num-1)
                        CGen_txt_right_send_size_$num++; // account for \n
                }
            }
            // send right
            if(__hps_node_id!=__hps_num_pes-1)
            {
                MPI_Isend(&CGen_txt_right_send_size_$num, 1, MPI_LONG_LONG_INT, __hps_node_id+1, 0, MPI_COMM_WORLD, &CGen_txt_MPI_request1_$num);
                MPI_Isend(CGen_txt_buffer_$num+CGen_txt_buff_size_$num-CGen_txt_right_send_size_$num, CGen_txt_right_send_size_$num, MPI_CHAR, __hps_node_id+1, 1, MPI_COMM_WORLD, &CGen_txt_MPI_request2_$num);
            }
            char* CGen_txt_left_buff_$num = NULL;
            int64_t CGen_txt_left_recv_size_$num = 0;
            // receive from left
            if(__hps_node_id!=0)
            {
                MPI_Recv(&CGen_txt_left_recv_size_$num, 1, MPI_LONG_LONG_INT, __hps_node_id-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                CGen_txt_left_buff_$num = new char[CGen_txt_left_recv_size_$num];
                MPI_Recv(CGen_txt_left_buff_$num, CGen_txt_left_recv_size_$num, MPI_CHAR, __hps_node_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // std::cout<<"rank: "<<__hps_node_id<<" received left "<<CGen_txt_left_recv_size_$num<<std::endl;
            }
            if(__hps_node_id!=__hps_num_pes-1)
            {
                MPI_Wait(&CGen_txt_MPI_request1_$num, &CGen_txt_MPI_status_$num);
                MPI_Wait(&CGen_txt_MPI_request2_$num, &CGen_txt_MPI_status_$num);
                // std::cout<<"rank: "<<__hps_node_id<<" sent right "<<CGen_txt_right_send_size_$num<<std::endl;
            }
            
            // int64_t total_data_size = (CGen_txt_end_$num-CGen_txt_start_$num)*CGen_txt_col_size_$num;
            // double *my_data = new double[total_data_size];
            int64_t CGen_txt_data_ind_$num = 0;
            
            char CGen_txt_sep_char_$num[] = \"\\n\";
            int64_t CGen_txt_curr_row_$num = 0;
            double* CGen_txt_data_arr = (double*)$arr.getData();
            while(CGen_txt_curr_row_$num!=CGen_txt_count_$num)
            {
                char* CGen_txt_line;
                if (CGen_txt_curr_row_$num==0)
                {
                    CGen_txt_line = strtok(CGen_txt_buffer_$num, CGen_txt_sep_char_$num);
                    if(CGen_txt_left_recv_size_$num!=0)
                    {
                        char *CGen_txt_tmp_line;
                        CGen_txt_tmp_line = new char[CGen_txt_left_recv_size_$num+strlen(CGen_txt_line)];
                        memcpy(CGen_txt_tmp_line, CGen_txt_left_buff_$num, CGen_txt_left_recv_size_$num);
                        memcpy(CGen_txt_tmp_line+CGen_txt_left_recv_size_$num, CGen_txt_line, strlen(CGen_txt_line));
                        CGen_txt_line = CGen_txt_tmp_line;
                    }
                }
                else if(CGen_txt_curr_row_$num==CGen_txt_count_$num-1)
                {
                    CGen_txt_line = strtok(NULL, CGen_txt_sep_char_$num);
                    if(CGen_txt_right_recv_size_$num!=0)
                    {
                        char *CGen_txt_tmp_line;
                        CGen_txt_tmp_line = new char[CGen_txt_right_recv_size_$num+strlen(CGen_txt_line)];
                        memcpy(CGen_txt_tmp_line, CGen_txt_line, strlen(CGen_txt_line));
                        memcpy(CGen_txt_tmp_line+strlen(CGen_txt_line), CGen_txt_right_buff_$num, CGen_txt_right_recv_size_$num);
                        CGen_txt_line = CGen_txt_tmp_line;
                    }
                }
                else
                {
                    CGen_txt_line = strtok(NULL, CGen_txt_sep_char_$num);
                }
                // parse line separately, not to change strtok's state
                for(int64_t i=0; i<CGen_txt_col_size_$num; i++)
                {
                    if(i==0)
                        CGen_txt_data_arr[CGen_txt_data_ind_$num++] = strtod(CGen_txt_line,&CGen_txt_line);
                    else
                        CGen_txt_data_arr[CGen_txt_data_ind_$num++] = strtod(CGen_txt_line+1,&CGen_txt_line);
         //           std::cout<<$arr[CGen_txt_data_ind_$num-1]<<std::endl;
                }
                CGen_txt_curr_row_$num++;
            }
            
            MPI_File_close(&dsrc_txt_file_$num);
            """
    end
    return s
end

function pattern_match_call_data_src_read(f::Any, v::Any, rf::Any, o::Any, arr::Any)
    return ""
end

function pattern_match_call_dist_h5_size(f::Symbol, size_arr::GenSym, ind::Union{Int64,SymAllGen})
    s = ""
    if f==:__hps_get_H5_dim_size || f==:__hps_get_TXT_dim_size
        dprintln(3,"match dist_dim_size ",f," ", size_arr, " ",ind)
        s = from_expr(size_arr)*"["*from_expr(ind)*"-1]"
    end
    return s
end

function pattern_match_call_dist_h5_size(f::Any, size_arr::Any, ind::Any)
    return ""
end

function pattern_match_call_kmeans(f::Symbol, cluster_out::SymAllGen, arr::SymAllGen, 
                                   num_clusters::SymAllGen, start::Symbol, count::Symbol, 
                                   col_size::Union{SymAllGen,Int,Expr}, tot_row_size::Union{SymAllGen,Int,Expr})
    s = ""
    if f==:__hps_kmeans
        c_arr = from_expr(arr)
        c_num_clusters = from_expr(num_clusters)
        c_col_size = from_expr(col_size)
        c_tot_row_size = from_expr(tot_row_size)
        c_cluster_out = from_expr(cluster_out)        
        
        s *= """
        byte   *nodeCentroids;
        size_t CentroidsArchLength;
        services::SharedPtr<NumericTable> centroids;
        InputDataArchive centroidsDataArch;
        int nIterations = 10;
        int mpi_root = 0;
        int rankId = __hps_node_id;

        HomogenNumericTable<double>* dataTable = new HomogenNumericTable<double>((double*)$c_arr.getData(), $c_col_size, $count);
        services::SharedPtr<NumericTable> dataTablePointer(dataTable);
        kmeans::init::Distributed<step1Local,double,kmeans::init::randomDense>
                       localInit($c_num_clusters, $c_tot_row_size, $start);
        localInit.input.set(kmeans::init::data, dataTablePointer);
        
        /* Compute k-means */
        localInit.compute();

        /* Serialize partial results required by step 2 */
        services::SharedPtr<byte> serializedData;
        InputDataArchive dataArch;
        localInit.getPartialResult()->serialize( dataArch );
        size_t perNodeArchLength = dataArch.getSizeOfArchive();

        /* Serialized data is of equal size on each node if each node called compute() equal number of times */
        if (rankId == mpi_root)
        {   
            serializedData = services::SharedPtr<byte>( new byte[ perNodeArchLength * __hps_num_pes ] );
        }   

        byte *nodeResults = new byte[ perNodeArchLength ];
        dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );

        /* Transfer partial results to step 2 on the root node */
        MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                MPI_COMM_WORLD);

        delete[] nodeResults;

        if(rankId == mpi_root)
        {   
            /* Create an algorithm to compute k-means on the master node */
            kmeans::init::Distributed<step2Master, double, kmeans::init::randomDense> masterInit($c_num_clusters);

            for( size_t i = 0; i < __hps_num_pes ; i++ )
            {   
                /* Deserialize partial results from step 1 */
                OutputDataArchive dataArch( serializedData.get() + perNodeArchLength * i, perNodeArchLength );

                services::SharedPtr<kmeans::init::PartialResult> dataForStep2FromStep1 = services::SharedPtr<kmeans::init::PartialResult>(
                                                                               new kmeans::init::PartialResult() );
                dataForStep2FromStep1->deserialize(dataArch);

                /* Set local partial results as input for the master-node algorithm */
                masterInit.input.add(kmeans::init::partialResults, dataForStep2FromStep1 );
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterInit.compute();
        masterInit.finalizeCompute();

        centroids = masterInit.getResult()->get(kmeans::init::centroids);
        }
        
        for(int iter=0; iter<nIterations; iter++) 
        {
        
            if(rankId == mpi_root)
            {
                /*Retrieve the algorithm results and serialize them */
                centroids->serialize( centroidsDataArch );
                CentroidsArchLength = centroidsDataArch.getSizeOfArchive();
            }

             /* Get partial results from the root node */
             MPI_Bcast( &CentroidsArchLength, sizeof(size_t), MPI_CHAR, mpi_root, MPI_COMM_WORLD );

              nodeCentroids = new byte[ CentroidsArchLength ];

            if(rankId == mpi_root)
            {
                centroidsDataArch.copyArchiveToArray( nodeCentroids, CentroidsArchLength );
            }

            MPI_Bcast( nodeCentroids, CentroidsArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD );

            /* Deserialize centroids data */
            OutputDataArchive centroidsDataArch( nodeCentroids, CentroidsArchLength );

            centroids = services::SharedPtr<NumericTable>( new HomogenNumericTable<double>() );

            centroids->deserialize(centroidsDataArch);

            /* Create an algorithm to compute k-means on local nodes */
            kmeans::Distributed<step1Local> localAlgorithm($c_num_clusters);

            /* Set the input data set to the algorithm */
            localAlgorithm.input.set(kmeans::data,           dataTablePointer);
            localAlgorithm.input.set(kmeans::inputCentroids, centroids);
    
            /* Compute k-means */
            localAlgorithm.compute();

            /* Serialize partial results required by step 2 */
            services::SharedPtr<byte> serializedData;
            InputDataArchive dataArch;
            localAlgorithm.getPartialResult()->serialize( dataArch );
            size_t perNodeArchLength = dataArch.getSizeOfArchive();

            /* Serialized data is of equal size on each node if each node called compute() equal number of times */
            if (rankId == mpi_root)
            {
                serializedData = services::SharedPtr<byte>( new byte[ perNodeArchLength * __hps_num_pes ] );
            }
            byte *nodeResults = new byte[ perNodeArchLength ];
            dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );

            /* Transfer partial results to step 2 on the root node */
            MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                        MPI_COMM_WORLD);

            delete[] nodeResults;

            if(rankId == mpi_root)
            {
               /* Create an algorithm to compute k-means on the master node */
               kmeans::Distributed<step2Master> masterAlgorithm($c_num_clusters);

               for( size_t i = 0; i < __hps_num_pes ; i++ )
                {
                    /* Deserialize partial results from step 1 */
                    OutputDataArchive dataArch( serializedData.get() + perNodeArchLength * i, perNodeArchLength );

                    services::SharedPtr<kmeans::PartialResult> dataForStep2FromStep1 = services::SharedPtr<kmeans::PartialResult>(
                                                                               new kmeans::PartialResult() );
                    dataForStep2FromStep1->deserialize(dataArch);

                    /* Set local partial results as input for the master-node algorithm */
                    masterAlgorithm.input.add(kmeans::partialResults, dataForStep2FromStep1 );
                }

                /* Merge and finalizeCompute k-means on the master node */
                masterAlgorithm.compute();
                masterAlgorithm.finalizeCompute();

                /* Retrieve the algorithm results */
                centroids = masterAlgorithm.getResult()->get(kmeans::centroids);
            }
            delete[] nodeCentroids;
        }
        
        BlockDescriptor<double> block;
        
        //std::cout<<centroids->getNumberOfRows()<<std::endl;
        //std::cout<<centroids->getNumberOfColumns()<<std::endl;
        
        centroids->getBlockOfRows(0, $c_num_clusters, readOnly, block);
        double* out_arr = block.getBlockPtr();
        //std::cout<<"output ";
        //for(int i=0; i<$c_col_size*$c_num_clusters; i++)
        //{
        //    std::cout<<" "<<out_arr[i];
        //}
        //std::cout<<std::endl;
        int64_t res_dims[] = {$c_col_size,$c_num_clusters};
        double* out_data = new double[$c_col_size*$c_num_clusters];
        memcpy(out_data, block.getBlockPtr(), $c_col_size*$c_num_clusters*sizeof(double));
        j2c_array<double> kmeans_out(out_data,2,res_dims);
        $c_cluster_out = kmeans_out;
    """
        
    end
    return s
end


function pattern_match_call_kmeans(f::ANY, cluster_out::ANY, arr::ANY, num_clusters::ANY, start::ANY, count::ANY, cols::ANY, rows::ANY)
    return ""
end

function pattern_match_call_linear_regression(f::Symbol, coeff_out::SymAllGen, points::SymAllGen, 
                                   responses::SymAllGen, start_points::Symbol, count_points::Symbol, 
                                   col_size_points::Union{SymAllGen,Int,Expr}, tot_row_size_points::Union{SymAllGen,Int,Expr},
                                   start_responses::Symbol, count_responses::Symbol, 
                                   col_size_responses::Union{SymAllGen,Int,Expr}, tot_row_size_responses::Union{SymAllGen,Int,Expr})
    s = ""
    if f==:__hps_LinearRegression
        c_points = from_expr(points)
        c_responses = from_expr(responses)
        c_col_size_points = from_expr(col_size_points)
        c_tot_row_size_points = from_expr(tot_row_size_points)
        c_col_size_responses = from_expr(col_size_responses)
        c_tot_row_size_responses = from_expr(tot_row_size_responses)
        c_coeff_out = from_expr(coeff_out)
        s = """
            assert($c_tot_row_size_points==$c_tot_row_size_responses);
            int mpi_root = 0;
            int rankId = __hps_node_id;
            
            HomogenNumericTable<double>* dataTable = new HomogenNumericTable<double>((double*)$c_points.getData(), $c_col_size_points, $count_points);
            HomogenNumericTable<double>* responseTable = new HomogenNumericTable<double>((double*)$c_responses.getData(), $c_col_size_responses, $count_responses);
            services::SharedPtr<NumericTable> trainData(dataTable);
            services::SharedPtr<NumericTable> trainDependentVariables(responseTable);
            services::SharedPtr<linear_regression::training::Result> trainingResult; 
        
            /* Create an algorithm object to train the multiple linear regression model based on the local-node data */
            linear_regression::training::Distributed<step1Local, double, linear_regression::training::qrDense> localAlgorithm;
        
            /* Pass a training data set and dependent values to the algorithm */
            localAlgorithm.input.set(linear_regression::training::data, trainData);
            localAlgorithm.input.set(linear_regression::training::dependentVariables, trainDependentVariables);
        
            /* Train the multiple linear regression model on local nodes */
            localAlgorithm.compute();
        
            /* Serialize partial results required by step 2 */
            services::SharedPtr<byte> serializedData;
            InputDataArchive dataArch;
            localAlgorithm.getPartialResult()->serialize( dataArch );
            size_t perNodeArchLength = dataArch.getSizeOfArchive();
        
            /* Serialized data is of equal size on each node if each node called compute() equal number of times */
            if (rankId == mpi_root)
            {
                serializedData = services::SharedPtr<byte>( new byte[ perNodeArchLength * __hps_num_pes] );
            }
        
            byte *nodeResults = new byte[ perNodeArchLength ];
            dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );
        
            /* Transfer partial results to step 2 on the root node */
            MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                        MPI_COMM_WORLD);
        
            delete[] nodeResults;
            services::SharedPtr<NumericTable> trainingCoeffsTable;
            if(rankId == mpi_root)
            {
                /* Create an algorithm object to build the final multiple linear regression model on the master node */
                linear_regression::training::Distributed<step2Master, double, linear_regression::training::qrDense> masterAlgorithm;
        
                for( size_t i = 0; i < __hps_num_pes; i++ )
                {
                    /* Deserialize partial results from step 1 */
                    OutputDataArchive dataArch( serializedData.get() + perNodeArchLength * i, perNodeArchLength );
        
                    services::SharedPtr<linear_regression::training::PartialResult> dataForStep2FromStep1 = services::SharedPtr<linear_regression::training::PartialResult>
                                                                               ( new linear_regression::training::PartialResult() );
                    dataForStep2FromStep1->deserialize(dataArch);
        
                    /* Set the local multiple linear regression model as input for the master-node algorithm */
                    masterAlgorithm.input.add(linear_regression::training::partialModels, dataForStep2FromStep1);
                }
        
                /* Merge and finalizeCompute the multiple linear regression model on the master node */
                masterAlgorithm.compute();
                masterAlgorithm.finalizeCompute();
        
                /* Retrieve the algorithm results */
                trainingResult = masterAlgorithm.getResult();
                // printNumericTable(trainingResult->get(linear_regression::training::model)->getBeta(), "Linear Regression coefficients:");
                trainingCoeffsTable = trainingResult->get(linear_regression::training::model)->getBeta();
            
                BlockDescriptor<double> block;
        
            //std::cout<<trainingCoeffsTable->getNumberOfRows()<<std::endl;
            //std::cout<<trainingCoeffsTable->getNumberOfColumns()<<std::endl;
        
                trainingCoeffsTable->getBlockOfRows(0, $c_col_size_responses, readOnly, block);
                double* out_arr = block.getBlockPtr();
            
            // assuming intercept is required
            int64_t coeff_size = $c_col_size_points+1;
            //std::cout<<"output ";
            //for(int i=0; i<coeff_size*$c_col_size_responses; i++)
            //{
            //    std::cout<<" "<<out_arr[i];
            //}
            //std::cout<<std::endl;
            
                int64_t res_dims[] = {coeff_size,$c_col_size_responses};
                double* out_data = new double[coeff_size*$c_col_size_responses];
                memcpy(out_data, block.getBlockPtr(), coeff_size*$c_col_size_responses*sizeof(double));
                j2c_array<double> linear_regression_out(out_data,2,res_dims);
                $c_coeff_out = linear_regression_out;
            }
            """
    end
    return s
end

function pattern_match_call_linear_regression(f::ANY, coeff_out::ANY, arr::ANY, num_clusters::ANY, 
          start::ANY, count::ANY, cols::ANY, rows::ANY, start2::ANY, count2::ANY, cols2::ANY, rows2::ANY)
    return ""
end

function pattern_match_call_naive_bayes(f::Symbol, coeff_out::SymAllGen, points::SymAllGen, 
                                   labels::SymAllGen, num_classes::Union{SymAllGen,Int,Expr}, start_points::Symbol, count_points::Symbol, 
                                   col_size_points::Union{SymAllGen,Int,Expr}, tot_row_size_points::Union{SymAllGen,Int,Expr},
                                   start_labels::Symbol, count_labels::Symbol, 
                                   col_size_labels::Union{SymAllGen,Int,Expr}, tot_row_size_labels::Union{SymAllGen,Int,Expr})
    s = ""
    if f==:__hps_NaiveBayes
        c_points = from_expr(points)
        c_labels = from_expr(labels)
        c_col_size_points = from_expr(col_size_points)
        c_tot_row_size_points = from_expr(tot_row_size_points)
        c_col_size_labels = from_expr(col_size_labels)
        c_tot_row_size_labels = from_expr(tot_row_size_labels)
        c_coeff_out = from_expr(coeff_out)
        c_num_classes = from_expr(num_classes)
        
        s = """
            assert($c_tot_row_size_points==$c_tot_row_size_labels);
            int mpi_root = 0;
            int rankId = __hps_node_id;
            
            HomogenNumericTable<double>* dataTable = new HomogenNumericTable<double>((double*)$c_points.getData(), $c_col_size_points, $count_points);
            HomogenNumericTable<double>* responseTable = new HomogenNumericTable<double>((double*)$c_labels.getData(), $c_col_size_labels, $count_labels);
            services::SharedPtr<NumericTable> trainData(dataTable);
            services::SharedPtr<NumericTable> trainGroundTruth(responseTable);
            services::SharedPtr<multinomial_naive_bayes::training::Result> trainingResult; 
        
        
            /* Create an algorithm object to train the Na__ve Bayes model based on the local-node data */
            multinomial_naive_bayes::training::Distributed<step1Local> localAlgorithm($c_num_classes);
        
            /* Pass a training data set and dependent values to the algorithm */
            localAlgorithm.input.set(multinomial_naive_bayes::classifier::training::data,   trainData);
            localAlgorithm.input.set(multinomial_naive_bayes::classifier::training::labels, trainGroundTruth);
        
            /* Train the Na__ve Bayes model on local nodes */
            localAlgorithm.compute();
        
            /* Serialize partial results required by step 2 */
            services::SharedPtr<byte> serializedData;
            InputDataArchive dataArch;
            localAlgorithm.getPartialResult()->serialize(dataArch);
            size_t perNodeArchLength = dataArch.getSizeOfArchive();
        
            /* Serialized data is of equal size on each node if each node called compute() equal number of times */
            if (rankId == mpi_root)
            {
                serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * __hps_num_pes]);
            }
        
            byte *nodeResults = new byte[perNodeArchLength];
            dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );
        
            /* Transfer partial results to step 2 on the root node */
            MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                        MPI_COMM_WORLD);
        
            delete[] nodeResults;
        
            if(rankId == mpi_root)
            {
                /* Create an algorithm object to build the final Na__ve Bayes model on the master node */
                multinomial_naive_bayes::training::Distributed<step2Master> masterAlgorithm($c_num_classes);
        
                for(size_t i = 0; i < __hps_num_pes ; i++)
                {
                    /* Deserialize partial results from step 1 */
                    OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);
        
                    services::SharedPtr<multinomial_naive_bayes::classifier::training::PartialResult> dataForStep2FromStep1 = services::SharedPtr<multinomial_naive_bayes::classifier::training::PartialResult>
                                                                               (new multinomial_naive_bayes::classifier::training::PartialResult());
                    dataForStep2FromStep1->deserialize(dataArch);
        
                    /* Set the local Na__ve Bayes model as input for the master-node algorithm */
                    masterAlgorithm.input.add(multinomial_naive_bayes::classifier::training::partialModels, dataForStep2FromStep1);
                }
        
                /* Merge and finalizeCompute the Na__ve Bayes model on the master node */
                masterAlgorithm.compute();
                masterAlgorithm.finalizeCompute();
        
                /* Retrieve the algorithm results */
                trainingResult = masterAlgorithm.getResult();
                
                trainingLogpTable = trainingResult->get(multinomial_naive_bayes::training::model)->getLogP();
                trainingLogThetaTable = trainingResult->get(multinomial_naive_bayes::training::model)->getLogTheta();
            
                BlockDescriptor<double> block1, block2;
            
                std::cout<<trainingLogThetaTable->getNumberOfRows()<<std::endl;
                std::cout<<trainingLogThetaTable->getNumberOfColumns()<<std::endl;
                std::cout<<trainingLogpTable->getNumberOfRows()<<std::endl;
                std::cout<<trainingLogpTable->getNumberOfColumns()<<std::endl;
        
                trainingLogpTable->getBlockOfRows(0, $c_num_class, readOnly, block1);
                trainingLogThetaTable->getBlockOfRows(0, $c_num_class, readOnly, block2);
                double* out_arr1 = block1.getBlockPtr();
                double* out_arr2 = block2.getBlockPtr();
            
            //std::cout<<"output ";
            //for(int i=0; i<coeff_size*$c_col_size_labels; i++)
            //{
            //    std::cout<<" "<<out_arr[i];
            //}
            //std::cout<<std::endl;
            
                int64_t res_dims[] = {$c_num_classes+1,$c_num_classes};
                double* out_data = new double[$c_num_classes*($c_num_classes+1)];
                memcpy(out_data, block1.getBlockPtr(), $c_num_classes*sizeof(double));
                memcpy(out_data+$c_num_classes, block2.getBlockPtr(), $c_num_classes*$c_num_classes*sizeof(double));
                j2c_array<double> naive_bayes_out(out_data,2,res_dims);
                $c_coeff_out = naive_bayes_out;
            }
            """
    end
    return s
end

function pattern_match_call_naive_bayes(f::ANY, coeff_out::ANY, arr::ANY, arr2::ANY, numClass::Any, 
          start::ANY, count::ANY, cols::ANY, rows::ANY, start2::ANY, count2::ANY, cols2::ANY, rows2::ANY)
    return ""
end

function pattern_match_call(ast::Array{Any, 1})
    dprintln(3,"pattern matching ",ast)
    s = ""
    if length(ast)==1
         s = pattern_match_call_dist_init(ast[1])
    end
    if(length(ast)==2)
        s = pattern_match_call_throw(ast[1],ast[2])
        s *= pattern_match_call_math(ast[1],ast[2])
    end
    if(length(ast)==4)
        s = pattern_match_call_dist_reduce(ast[1],ast[2],ast[3], ast[4])
        # text file read
        s *= pattern_match_call_data_src_open(ast[1],ast[2],ast[3], ast[4])
    end
    if(length(ast)==5)
        # HDF5 open
        s = pattern_match_call_data_src_open(ast[1],ast[2],ast[3], ast[4], ast[5])
        s *= pattern_match_call_data_src_read(ast[1],ast[2],ast[3], ast[4], ast[5])
    end
    if(length(ast)==3) # randn! call has 3 args
        s = pattern_match_call_dist_h5_size(ast[1],ast[2],ast[3])
        s *= pattern_match_call_randn(ast[1],ast[2],ast[3])
        #sa*= pattern_match_call_powersq(ast[1],ast[2], ast[3])
        s *= pattern_match_call_reshape(ast[1],ast[2],ast[3])
    end
    if(length(ast)>=2) # rand! has 2 or more args
        s *= pattern_match_call_rand(ast...)
    end
    # gemm calls have 6 args
    if(length(ast)==6)
        s = pattern_match_call_gemm(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6])
    end
    if(length(ast)==8)
        s = pattern_match_call_kmeans(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8])
    end
    if(length(ast)==12)
        s = pattern_match_call_linear_regression(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12])
    end
    if(length(ast)==13)
        s = pattern_match_call_naive_bayes(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13])
    end
    return s
end


function from_assignment_match_hvcat(lhs, rhs::Expr)
    s = ""
    # if this is a hvcat call, the array should be allocated and initialized
    if rhs.head==:call && (checkTopNodeName(rhs.args[1],:typed_hvcat) || checkGlobalRefName(rhs.args[1],:hvcat))
        dprintln(3,"Found hvcat assignment: ", lhs," ", rhs)

        is_typed::Bool = checkTopNodeName(rhs.args[1],:typed_hvcat)
        
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
            arr_var = toSymGen(lhs)
            atyp, arr_dims = parseArrayType(lstate.symboltable[arr_var])
            typ = toCtype(atyp)
        end

        nr = length(rows)
        nc = rows[1] # all rows should have the same size
        s *= from_expr(lhs) * " = j2c_array<$typ>::new_j2c_array_2d(NULL, $nr, $nc);\n"
        s *= mapfoldl((i) -> from_setindex([lhs,values[i],convert(Int64,ceil(i/nr)),(i-1)%nr+1])*";", (a, b) -> "$a $b", 1:length(values))
    end
    return s
end

function from_assignment_match_hvcat(lhs, rhs::ANY)
    return ""
end

function from_assignment_match_cat_t(lhs, rhs::Expr)
    s = ""
    if rhs.head==:call && isa(rhs.args[1],GlobalRef) && rhs.args[1].name==:cat_t
        dims = rhs.args[2]
        @assert dims==2 "CGen: only 2d cat_t() is supported now"
        size = length(rhs.args[4:end])
        typ = toCtype(eval(rhs.args[3].name))
        s *= from_expr(lhs) * " = j2c_array<$typ>::new_j2c_array_$(dims)d(NULL, 1,$size);\n"
        values = rhs.args[4:end]
        s *= mapfoldl((i) -> from_setindex([lhs,values[i],i])*";", (a, b) -> "$a $b", 1:length(values))
    end
    return s
end

function from_assignment_match_cat_t(lhs, rhs::ANY)
    return ""
end

function from_assignment_match_dist(lhs::Symbol, rhs::Expr)
    dprintln(3, "assignment pattern match dist ",lhs," = ",rhs)
    if rhs.head==:call && length(rhs.args)==1 && isTopNode(rhs.args[1])
        dist_call = rhs.args[1].name
        if dist_call ==:hps_dist_num_pes
            return "MPI_Comm_size(MPI_COMM_WORLD,&$lhs);"
        elseif dist_call ==:hps_dist_node_id
            return "MPI_Comm_rank(MPI_COMM_WORLD,&$lhs);"
        end
    end
    return ""
end

function from_assignment_match_dist(lhs::GenSym, rhs::Expr)
    dprintln(3, "assignment pattern match dist2: ",lhs," = ",rhs)
    s = ""
    local num::AbstractString
    if rhs.head==:call && rhs.args[1]==:__hps_data_source_HDF5_size
        num = from_expr(rhs.args[2].id)
        s = "hid_t space_id_$num = H5Dget_space(dataset_id_$num);\n"    
        s *= "assert(space_id_$num != -1);\n"    
        s *= "hsize_t data_ndim_$num = H5Sget_simple_extent_ndims(space_id_$num);\n"
        s *= "hsize_t space_dims_$num[data_ndim_$num];\n"    
        s *= "H5Sget_simple_extent_dims(space_id_$num, space_dims_$num, NULL);\n"
        s *= from_expr(lhs)*" = space_dims_$num;"
    elseif rhs.head==:call && rhs.args[1]==:__hps_data_source_TXT_size
        num = from_expr(rhs.args[2].id)
        c_lhs = from_expr(lhs)
        s = """
            MPI_Offset CGen_txt_tot_file_size_$num;
            MPI_Offset CGen_txt_buff_size_$num;
            MPI_Offset CGen_txt_offset_start_$num;
            MPI_Offset CGen_txt_offset_end_$num;
        
            /* divide file read */
            MPI_File_get_size(dsrc_txt_file_$num, &CGen_txt_tot_file_size_$num);
            CGen_txt_buff_size_$num = CGen_txt_tot_file_size_$num/__hps_num_pes;
            CGen_txt_offset_start_$num = __hps_node_id * CGen_txt_buff_size_$num;
            CGen_txt_offset_end_$num   = CGen_txt_offset_start_$num + CGen_txt_buff_size_$num - 1;
            if (__hps_node_id == __hps_num_pes-1)
                CGen_txt_offset_end_$num = CGen_txt_tot_file_size_$num;
            CGen_txt_buff_size_$num =  CGen_txt_offset_end_$num - CGen_txt_offset_start_$num + 1;
        
            char* CGen_txt_buffer_$num = new char[CGen_txt_buff_size_$num+1];
        
            MPI_File_read_at_all(dsrc_txt_file_$num, CGen_txt_offset_start_$num, CGen_txt_buffer_$num, CGen_txt_buff_size_$num, MPI_CHAR, MPI_STATUS_IGNORE);
            CGen_txt_buffer_$num[CGen_txt_buff_size_$num] = \'\\0\';
            
            // make sure new line is there for last line
            if(__hps_node_id == __hps_num_pes-1 && CGen_txt_buffer_$num[CGen_txt_buff_size_$num-2]!=\'\\n\') 
                CGen_txt_buffer_$num[CGen_txt_buff_size_$num-1]=\'\\n\';
            
            // count number of new lines
            int64_t CGen_txt_num_lines_$num = 0;
            int64_t CGen_txt_char_index_$num = 0;
            while (CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\0\') {
                if(CGen_txt_buffer_$num[CGen_txt_char_index_$num]==\'\\n\')
                    CGen_txt_num_lines_$num++;
                CGen_txt_char_index_$num++;
            }
        
            // std::cout<<"rank: "<<__hps_node_id<<" lines: "<<CGen_txt_num_lines_$num<<" startChar: "<<CGen_txt_buffer_$num[0]<<std::endl;
            // get total number of rows
            int64_t CGen_txt_tot_row_size_$num=0;
            MPI_Allreduce(&CGen_txt_num_lines_$num, &CGen_txt_tot_row_size_$num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            // std::cout<<"total rows: "<<CGen_txt_tot_row_size_$num<<std::endl;
            
            // count number of values in a column
            // 1D data has CGen_txt_col_size_$num==1
            int64_t CGen_txt_col_size_$num = 1;
            CGen_txt_char_index_$num = 0;
            while (CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\0\' && CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\n\')
                CGen_txt_char_index_$num++;
            CGen_txt_char_index_$num++;
            while (CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\0\' && CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\n\') {
                if(CGen_txt_buffer_$num[CGen_txt_char_index_$num]==',')
                    CGen_txt_col_size_$num++;
                CGen_txt_char_index_$num++;
            }
            
            // prefix sum to find current global starting line on this node
            int64_t CGen_txt_curr_start_$num = 0;
            MPI_Scan(&CGen_txt_num_lines_$num, &CGen_txt_curr_start_$num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            int64_t CGen_txt_curr_end_$num = CGen_txt_curr_start_$num;
            CGen_txt_curr_start_$num -= CGen_txt_num_lines_$num; // Scan is inclusive
            if(CGen_txt_col_size_$num==1) {
                $c_lhs = new uint64_t[1];
                $c_lhs[0] = CGen_txt_tot_row_size_$num;
            } else {
                $c_lhs = new uint64_t[2];
                $c_lhs[0] = CGen_txt_tot_row_size_$num;
                $c_lhs[1] = CGen_txt_col_size_$num;
            }
            """
    end
    return s
end

function isTopNode(a::TopNode)
    return true
end

function isTopNode(a::Any)
    return false
end

function from_assignment_match_dist(lhs::Any, rhs::Any)
    return ""
end

