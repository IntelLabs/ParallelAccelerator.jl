#include <cstdio>
#include <cstdlib>
#include <vector>
using std::vector;
#include <mpi.h>
#include "Interface.h"
#include <cassert>

#include "DArray.h"
#include "DataMap.h"
#include "IterMap.h"
#include "pse-runtime.h"
#include "DataRequest.h"

void Interface::InsertDivisibleTask (char *_task_name, /*name of the task */
		pert_range_Nd_t _iterations,  /* iteration range with dimension infor */
		unsigned int _num_args, /*number of arguments*/
		void **_args, /*array of argument information*/
		pert_arg_metadata_t *_args_metadata, /*array of metadata for arguments */
		void (_host_par_func)(pert_range_Nd_t , void **), /* Function pointer to Xeon host data-parallel function binary */
		void (_phi_par_func)(pert_range_Nd_t , void **, pert_arg_offset_t *, /*char *mic_1, char *mic_2, int64_t *offset_mic, unsigned int pipeline_stage,*/ pert_double_buffer_t dbuffer_metadata, char *, unsigned int mic_id),  /* Function pointer to PHI data-parallel function binary */
		void (_host_join_func)(void **, void **), /* Function pointer to Xeon host reduction function binary */
		void (_phi_map_func)(void *, void *, int64_t num_bytes), /* A map operation for the output variables coming from  phi*/
		void (_phi_wait_func)(char *, unsigned int),
		pert_task_options_t _task_opts, /* Task options: priority, scheduling hints, sequential task or not, implicit completion, ... */
		void **_deallocate_args, /* array of pointers to be used for deallocation */
		void (_deallocate)(void **), /* deallocation callback for reclaiming memory after task is finished */
		pert_grain_size_t *_host_grain_size, /* minimum grain_size for basic tasks */
		pert_grain_size_t *_phi_grain_size, /* minimum grain_size for basic tasks */
		void *_sched_hints_handle   /* handle to the LDL profiling information */
		)
{
	task_name = _task_name;
	iterations = _iterations;
	num_args = _num_args;
	args = _args;
	args_metadata = _args_metadata;
	host_par_func = _host_par_func;
	phi_par_func = _phi_par_func;
	host_join_func = _host_join_func;
	phi_map_func = _phi_map_func;
	phi_wait_func = _phi_wait_func;
	task_opts = _task_opts;
	deallocate_args = _deallocate_args;
	deallocate = _deallocate;
	host_grain_size = _host_grain_size;
	phi_grain_size = _phi_grain_size;
	sched_hints_handle = _sched_hints_handle;

/*	uint64_t args_len = (uint64_t)args[0];
	uintptr_t* _args = (uintptr_t*)args;
	pert_arg_queue_t args_queue((void**)_args, args_len);
	pert_arg_helper args_helper(&args_queue);
	args_helper.read((uint64_t)0, VALUE);
	args_helper.read((uintptr_t)0, VALUE);
	int64_t &_parallel_ir_save_array_len_1_1 = *args_helper.read((int64_t)0, VALUE);
	j2c_array<double>  _A;
	args_helper.read(_A, IN);
	j2c_array<double>  _parallel_ir_new_array_name_1_1;
	args_helper.read(_parallel_ir_new_array_name_1_1, OUT);
*/
	int numLocalTasks = 0;
	IterMap** argIterMaps = new IterMap*[num_args];
	vector<DataRequest*>** argOutReqs = new vector<DataRequest*>*[num_args];
	vector<DataRequest*>** argInReqs = new vector<DataRequest*>*[num_args];
	

	for(int i=0; i<num_args; i++)
	{
		if( args_metadata[i].is_scalar || !( args_metadata[i].arg_opt == IN || args_metadata[i].arg_opt == OUT || args_metadata[i].arg_opt == INOUT ) )
			continue;
		// TODO why one element arrays?
		if(args_metadata[i].array_access_desc->dim==1 && args_metadata[i].array_access_desc->max_size[0]==1)
			continue;

		// TODO hack for now to distribute arrays
		args_metadata[i].d_arr_id = distributeArray(i);

		DArray* A = DArray::getArr(args_metadata[i].d_arr_id);
		
		// create iteration map object to map iterations to nodes
		argIterMaps[i]= new IterMap(args_metadata[i].array_access_desc, &iterations, 0);

		numLocalTasks = argIterMaps[i]->numLocalTasks();

//		if(rank==print_rank)
//			argIterMaps[i]->print();

		// create data requests for data that needs to be sent out
		vector<DataRequest*> *outReqsA = A->getOutReqs(argIterMaps[i]);
		argOutReqs[i] = outReqsA;
//		if(rank==print_rank)
//		{
//			printf("A out reqs (%d): ",outReqsA->size());
//			for(int i=0; i<outReqsA->size(); i++)
//				outReqsA->at(i)->print();
//			printf("\n");
//		}


		// create data requests for data that needs to be received
		vector<DataRequest*> *inReqsA = A->getInReqs(argIterMaps[i]);
		argInReqs[i] = inReqsA;

//		if(rank==print_rank)
//		{
//			printf("A in reqs (%d): ",inReqsA->size());
//			for(int i=0; i<inReqsA->size(); i++)
//				inReqsA->at(i)->print();
//			printf("\n");
//		}

		// post asynchronous send and receives (MPI_Isend and MPI_Irecv)
		DataRequest::startAllComms(outReqsA);
		DataRequest::startAllComms(inReqsA);
	}

	vector<Task> allTasks;

	for(int t=0; t< numLocalTasks; t++)
	{
		// global arrays
		vector<DArray*> darrays;
		vector<Block> accessBlocks;
		vector<vector<DataRequest*>*> allInReqs;
		vector<vector<DataRequest*>*> allOutReqs;
		IterMap* myMap;
		
		for(int i=0; i<num_args; i++)
		{
			if( args_metadata[i].is_scalar || !( args_metadata[i].arg_opt == IN || args_metadata[i].arg_opt == OUT || args_metadata[i].arg_opt == INOUT ) )
				continue;
			// TODO why one element arrays?
			if(args_metadata[i].array_access_desc->dim==1 && args_metadata[i].array_access_desc->max_size[0]==1)
				continue;

			DArray* A = DArray::getArr(args_metadata[i].d_arr_id);
			darrays.push_back(A);
			// access blocks for each arg
			IterMap* im_A = argIterMaps[i];
			myMap = im_A;

			Block A_access = im_A->getMyAccessBlock(t);
			accessBlocks.push_back(A_access);

			vector<DataRequest*> *inReqsA = argInReqs[i];
			vector<DataRequest*> *outReqsA = argOutReqs[i];
			// data requests for each arg (in,out)
			vector<DataRequest*> *taskInReqsA = DataRequest::getDataReqsForTask(inReqsA, t);
			vector<DataRequest*> *taskOutReqsA = DataRequest::getDataReqsForTask(outReqsA, t);
			allInReqs.push_back(taskInReqsA);
			allOutReqs.push_back(taskOutReqsA);
		}

		allTasks.push_back(Task(myMap->getMyIterBlock(t),
					darrays,
					accessBlocks,
					allInReqs, allOutReqs,
					this));
	}
	
	// for each task
	for(int i=0; i< allTasks.size(); i++)
	{
		// wait for data to arrive
		allTasks[i].waitForData();
		allTasks[i].execute();
		allTasks[i].writeBack();
	}

	// TODO hack: gathering all data on all nodes for now (1D only)
	for(int i=0; i<num_args; i++)
	{
		if( args_metadata[i].is_scalar || !( args_metadata[i].arg_opt == OUT || args_metadata[i].arg_opt == INOUT ) )
			continue;
		if(args_metadata[i].array_access_desc->dim==1 && args_metadata[i].array_access_desc->max_size[0]==1)
			continue;
		 DArray* A = DArray::getArr(args_metadata[i].d_arr_id);
		 // only one block is supported
		 assert(A->getNumBlocks()==1);
		 // can't handle inout yet
		 assert(args_metadata[i].arg_opt != INOUT);

		// void* localData = A->getDataBlock(0);

		int localSize = A->getBlockSize(0)*A->getElemSize();
		void* localData = new char[localSize];
		void* start = args[i]+A->getBlockStart1D(0);
		memcpy(localData, start, localSize);
		MPI_Allgather(localData, localSize, MPI_CHAR, args[i], localSize, MPI_CHAR, MPI_COMM_WORLD);
	}
}


int Interface::distributeArray(int argNum)
{
	pert_array_access_desc_Nd_t * arr_desc = args_metadata[argNum].array_access_desc;
	int dim = arr_desc->dim;
	int *dim_sizes = new int[dim];
	for(int i=0; i<dim; i++) 
		dim_sizes[i] = arr_desc->max_size[i];
	// create data map object to map data of arrays to nodes
	// zero for second argument means the runtime decides block size automatically
	DataMap *d_map_A = new DataMap(dim, 0, dim_sizes);
	DArray *A = new DArray(d_map_A, args_metadata[argNum].size_of_each_element_in_bytes);

//	if(rank==print_rank)
//		d_map_A.print();

// TODO: for ouput arrays, only allocate
	if(args_metadata[argNum].arg_opt == IN || args_metadata[argNum].arg_opt == INOUT)
		A->getMyData(args[argNum]);

	return A->getID();
}

// TODO
void Interface::setArray(void *data, int i)
{
}

void Interface::runTask(Block iterBlock)
{
	pert_range_Nd_t range = iterBlock.toRangeT();
	host_par_func(range, args);
}



extern "C" void InitializeTiming()
{
}

extern "C"
error_t pert_init (bool pre_allocate_mic_buffers)
{
	MPI_Init(0,NULL);
	return 0;
}

extern "C"
error_t pert_shutdown ()
{
	MPI_Finalize();
	return 0;
}
extern "C" void FinalizeTiming()
{
}



  /* Divisible Task insertion API -- useful for data-parallel programming
     Each function is declared in the following way:
     void host_func(pert_range_Nd_t *, void **packed_data_wrapper);
     The first thing the host_func does is to parse the arguments from packed_data_wrapper. 
     */
  error_t pert_insert_divisible_task (char *task_name, /*name of the task */
                                      pert_range_Nd_t iterations,  /* iteration range with dimension infor */
                                      unsigned int num_args, /*number of arguments*/
                                      void **args, /*array of argument information*/
                                      pert_arg_metadata_t *args_metadata, /*array of metadata for arguments */
                                      void (host_par_func)(pert_range_Nd_t , void **), /* Function pointer to Xeon host data-parallel function binary */
                                      void (phi_par_func)(pert_range_Nd_t , void **, pert_arg_offset_t *, /*char *mic_1, char *mic_2, int64_t *offset_mic, unsigned int pipeline_stage,*/ pert_double_buffer_t dbuffer_metadata, char *, unsigned int mic_id),  /* Function pointer to PHI data-parallel function binary */
                                      void (host_join_func)(void **, void **), /* Function pointer to Xeon host reduction function binary */
                                      void (phi_map_func)(void *, void *, int64_t num_bytes), /* A map operation for the output variables coming from  phi*/
                                      void (phi_wait_func)(char *, unsigned int),
                                      pert_task_options_t task_opts, /* Task options: priority, scheduling hints, sequential task or not, implicit completion, ... */
                                      void **deallocate_args, /* array of pointers to be used for deallocation */
                                      void (deallocate)(void **), /* deallocation callback for reclaiming memory after task is finished */
                                      pert_grain_size_t *host_grain_size, /* minimum grain_size for basic tasks */
                                      pert_grain_size_t *phi_grain_size, /* minimum grain_size for basic tasks */
                                      void *sched_hints_handle   /* handle to the LDL profiling information */
                                     )
{
	Interface interface;
	interface.InsertDivisibleTask(task_name,iterations,num_args,args,args_metadata,host_par_func,phi_par_func,host_join_func,phi_map_func,phi_wait_func,task_opts,deallocate_args,deallocate,host_grain_size,phi_grain_size,sched_hints_handle);
	return 0;
}



extern "C" void initDArray(int* dims, int num_dims, int* id)
{
	id = 0;
	return;
}


extern "C" void setPtr(void* array, int id)
{
	return;
}



