#####################################################
#####################################################
#
#	A PoC illustrating a hypothetical extension to |cartesianarray| that
#	allows executing portions of the iteration space on different nodes in a cluster.
#	For example use, see the "render" function at the bottom. Briefly, we introduce a 
#	|distribute| macro that takes a cartesianarray operation as argument.
#
#		a = @distribute cartesianarray((i, j) -> i*j, Int64, (w,h))
#		a = @distribute cartesianarray((i) -> i*2, Int64, w)
#
#	Also see below a simple matmul example using this macro.
#	To actually use it, set up a key-based (password-less) ssh access on the
#	psephi machines, then run julia with a machinefile that contains the 
#	psephi hosts.
#	
#####################################################
#####################################################

require("intel-pse.jl")
importall ParallelAccelerator
ParallelIR.set_debug_level(3)
ParallelIR.PIRSetFuseLimit(-1)
ParallelAccelerator.set_debug_level(3)
ccall(:set_j2c_verbose, Void, (Cint,), 9)

# A variant of pmap that passes indexes instead of values
# For now, we don't use this.
function dpmap(f, lsts...; err_retry=true, err_stop=false, pids = workers())
    len = length(lsts)

    results = Dict{Int,Any}()

    retryqueue = []
    task_in_err = false
    is_task_in_error() = task_in_err
    set_task_in_error() = (task_in_err = true)

    nextidx = 0
    getnextidx() = (nextidx += 1)

    states = [start(lsts[idx]) for idx in 1:len]
    function getnext_tasklet()
        if is_task_in_error() && err_stop
            return nothing
        elseif !any(idx->done(lsts[idx],states[idx]), 1:len)
            nxts = [next(lsts[idx],states[idx]) for idx in 1:len]
            for idx in 1:len; states[idx] = nxts[idx][2]; end
            nxtvals = [x[1] for x in nxts]
            return (getnextidx(), nxtvals)
        elseif !isempty(retryqueue)
            return shift!(retryqueue)
        else
            return nothing
        end
    end

    @sync begin
        for wpid in pids
            @async begin
                tasklet = getnext_tasklet()
                while (tasklet != nothing)
                    (idx, fvals) = tasklet
                    try
						# Instead of passing value as in pmap,
						# we want to pass idx
                        #result = remotecall_fetch(wpid, f, fvals...)
                        result = remotecall_fetch(wpid, f, idx)
                        if isa(result, Exception)
                            ((wpid == myid()) ? rethrow(result) : throw(result))
                        else
                            results[idx] = result
                        end
                    catch ex
                        if err_retry
                            push!(retryqueue, (idx,fvals, ex))
                        else
                            results[idx] = ex
                        end
                        set_task_in_error()
                        break # remove this worker from accepting any more tasks
                    end

                    tasklet = getnext_tasklet()
                end
            end
        end
    end

    for failure in retryqueue
        results[failure[1]] = failure[3]
    end
    [results[x] for x in 1:nextidx]
end

function distribute1DF(body, T, ndims)
	_f = (function foo(_pid, _csize)
		a = Array(T, _csize)
		cartesianarray(
			function _inner(idx)
				a[idx] = body((idx+((_pid-1)*_csize)))
			end,
			T, (_csize,))
		a
		end)
	c = Array(T, ndims)
	tileSize = int(floor((ndims/nprocs())))
	@sync begin
		for pid in 1:nprocs()
			@async begin
        		result = remotecall_fetch(pid, _f, pid, tileSize)
				copy!(c, (pid-1)*tileSize+1, result, 1, tileSize)
			end
		end
		for i in tileSize*nprocs()+1:ndims
			c[i] = body(i) 
		end
	end
	c
end

function distribute2DF(body, T, ndims)
	_f = (function foo(_pid, _csize)
		#ParallelAccelerator.offload(_f, (Int64, Int64))
		#println("[DEBUG] On process: ", myid(), " doing chunk of size: ", _csize)
		a = Array(T, _csize)
		cartesianarray(
			function _inner(idx1, idx2)
				a[idx1, idx2] = body((idx1+((_pid-1)*_csize[1])), idx2)
			end,
			T, _csize)
		#println("[DEBUG] On process: ", myid(), " tilesize is: ", _csize)
		#println("[DEBUG] On process: ", myid(), " done with compute")
		a
		end)
	c = Array(T, ndims)
	tileSize = int(floor((ndims[1]/nprocs())))
	#println("[DEBUG] On process: ", myid(), " tilesize is: ", tileSize)
	@sync begin
		for pid in nprocs():-1:1
			@async begin
				#println("Spawning process: ", pid)
        		result = remotecall_fetch(pid, _f, pid, (tileSize, ndims[2]))
				for i in 1:tileSize
					#copy!(c[(pid-1)*tileSize+i], result, 1, tileSize)
					for j in 1:ndims[2]
						c[(pid-1)*tileSize+i, j] = result[i, j]
					end
				end
			end
		end

		#println("Done with compute, ", tileSize*nprocs()+1, ":", ndims[1], ":", ndims[2])
		#for i in tileSize*nprocs()+1:ndims[1]
		#	for j in 1:ndims[2]
		#		c[i, j] = body(i, j)
		#	end
		#end
	end
	c
end

macro distribute1D(cm)
	return :(distribute1DF($(cm.args[2]), $(cm.args[3]), $(cm.args[4])))
end

macro distribute2D(cm)
	return :(distribute2DF($(cm.args[2]), $(cm.args[3]), $(cm.args[4])))
end

function rank(crtOp)
	if isa(((crtOp.args[4])), Symbol)
		return 1
	else
	    return (length((crtOp.args[4].args)))
	end
end

macro distribute(crtOp)
	@printf("[DEBUG] Using %d processes\n", nprocs())
    (isa(crtOp, Expr) && 
		crtOp.args[1] == :cartesianarray) || 
			throw("Can only distribute cartesianarray operations")
    cRank = rank(crtOp)
    if(cRank == 1)
        return :(@distribute1D($crtOp))
    elseif(cRank == 2)
        return :(@distribute2D($crtOp))
    else
		println("Unknown rank, switching to sequential execution", cRank)
        return crtOp
    end
end

function matmul(A::Array{Float64,2}, B::Array{Float64,2}, si::Int64)
    uDim = size(A)[2]
	C = @distribute cartesianarray(function(i, j)
    	    sum = 0.0
        	for k = 1:uDim
	        	sum += A[i,k]*B[k,j]
        	end
        	sum
        	end,
        Float64, (size(A)[1], size(B)[2]))
    C
end
function render(w::Int64, h::Int64)
	# Example usages
	#a = @distribute cartesianarray((i, j) -> i*j, Int64, (w,h))
	#a = @distribute cartesianarray((i) -> i*2, Int64, w)
	a = matmul(ones(Float64, w, h), ones(Float64, h, w), 0)
	#a
end
#ParallelAccelerator.offload(render, (Int64, Int64))
function main()
	w = 1024
	h = 1024
	c = @time render(w, h)
	println(sum(c))
end
main()
