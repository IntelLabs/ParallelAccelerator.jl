#include <cstdio>
#include <cstdlib>
#include <vector>
using std::vector;

#include "DArray.h"
#include "DataMap.h"
#include "IterMap.h"
#include "pse-runtime.h"
#include "DataRequest.h"
#include <mpi.h>

void func(Block iterBlock, void* A, void*C) {}


void initIterRange(pert_array_access_desc_Nd_t *accessRangeA, pert_array_access_desc_Nd_t *accessRangeC, pert_range_Nd_t *iterRange, int* dim_sizes, int dim);

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	// dimensions and sizes
	int dim = 1;
	int dim_size = 1000;
	int print_rank=1;
	if(argc==4)
	{
		dim=atoi(argv[1]);
		dim_size=atoi(argv[2]);
		print_rank = atoi(argv[3]);
	}
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank==0) 
		printf("running with dim:%d dim_size:%d print_rank:%d\n",dim, dim_size, print_rank);
	/* example runtime code executed for C = 2.*A */

	int *dim_sizes = new int[dim];
	for(int i=0; i<dim; i++)
		dim_sizes[i] = dim_size;

	// create data map object to map data of arrays to nodes
	// zero for second argument means the runtime decides block size automatically
	DataMap d_map_A(dim, 0, dim_sizes);
	DataMap d_map_C(dim, 0, dim_sizes);

	// create arrays using the map object
	DArray A(&d_map_A, sizeof(double));
	DArray C(&d_map_C, sizeof(double));
	if(rank==print_rank)
		d_map_A.print();

	// data access info provided by PSE compiler for A and C
	pert_array_access_desc_Nd_t accessRangeA;
	pert_array_access_desc_Nd_t accessRangeC;
	pert_range_Nd_t iterRange;
	initIterRange(&accessRangeA, &accessRangeC, &iterRange, dim_sizes, dim);

	// create iteration map object to map iterations to nodes
	IterMap im_A(&accessRangeA, &iterRange, 0);
	IterMap im_C(&accessRangeC, &iterRange, 0);
	if(rank==print_rank)
	im_A.print();
	// create data requests for data that needs to be sent out
	vector<DataRequest*> *outReqsA = A.getOutReqs(&im_A);
	outReqsA->at(0)->print();
	vector<DataRequest*> *outReqsC = C.getOutReqs(&im_C);

	if(rank==print_rank)
	{
		printf("A out reqs (%d): ",outReqsA->size());
		for(int i=0; i<outReqsA->size(); i++)
			outReqsA->at(i)->print();
		printf("\n");
	}


	// create data requests for data that needs to be received
	vector<DataRequest*> *inReqsA = A.getInReqs(&im_A);
	vector<DataRequest*> *inReqsC = C.getInReqs(&im_C);
	
	if(rank==print_rank)
	{
		printf("A in reqs (%d): ",inReqsA->size());
		for(int i=0; i<inReqsA->size(); i++)
			inReqsA->at(i)->print();
		printf("\n");
	}

	// post asynchronous send and receives (MPI_Isend and MPI_Irecv)
	DataRequest::startAllComms(outReqsA);
	DataRequest::startAllComms(inReqsA);

	// post receives for C
	DataRequest::startAllComms(inReqsC);

	vector<Task> allTasks;
	for(int i=0; i< im_A.numLocalTasks(); i++)
	{
		// global arrays
		vector<DArray*> darrays;
		darrays.push_back(&A);
		darrays.push_back(&C);
		// access blocks for each arg
		Block A_access = im_A.getMyAccessBlock(i);
		Block C_access = im_C.getMyAccessBlock(i);
		vector<Block> accessBlocks;
		accessBlocks.push_back(A_access);
		accessBlocks.push_back(C_access);

		// data requests for each arg (in,out)
		vector<DataRequest*> *taskInReqsA = DataRequest::getDataReqsForTask(inReqsA, i);
		vector<DataRequest*> *taskInReqsC = NULL; // Output array, no in requests 
		vector<DataRequest*> *taskOutReqsA = DataRequest::getDataReqsForTask(outReqsA, i);
		vector<DataRequest*> *taskOutReqsC = DataRequest::getDataReqsForTask(outReqsC, i);
		vector<vector<DataRequest*>*> allInReqs;
		allInReqs.push_back(taskInReqsA);
		allInReqs.push_back(taskInReqsC);
		vector<vector<DataRequest*>*> allOutReqs;
		allOutReqs.push_back(taskOutReqsA);
		allOutReqs.push_back(taskOutReqsC);

/*		allTasks.push_back(Task(im_A.getMyIterBlock(i),
					darrays,
					accessBlocks,
					allInReqs, allOutReqs,
					func));
*/	}
	// create local task info from iteration maps
//	vector<IterMap*> allIterMaps;
//	iterMaps.push_back(im_A);
//	iterMaps.push_back(im_C);

//	vector<vector<DataRequest*>*> allInReqs;
//	allInReqs.push_back();

//	vector<Task> tasks = Task::createLocalTasks(im_A, im_C);
//	vector<Task> tasks = Task::createLocalTasks(im_A, im_C);
//	vector<Task> tasks = IterMap::createLocalTasks(im_A, im_C);

	// for each task
//	for(int i=0; i< tasks.size(); i++)
	for(int i=0; i< allTasks.size(); i++)
	{
		// wait for data to arrive
		allTasks[i].waitForData();
		allTasks[i].execute();
		allTasks[i].writeBack();
		// construct data in local pointer
	//	double* localA = A.getLocalPointer(taskReqs);
		// get local pointer for C; no need to wait since it is in write mode
	//	double* localC = C.getLocalPointer(&im_C, i);

//		tasks[i].setLocPointer(A, localA);
//		tasks[i].setLocPointer(C, localC);
		// execute the local task
//		tasks[i].execTask();
		// write back C data to distributed array
//		C.writeBack(outReqsC, i, localC);
	}
	return 0;
}


void initIterRange(pert_array_access_desc_Nd_t *accessRangeA, pert_array_access_desc_Nd_t *accessRangeC, pert_range_Nd_t *iterRange, int* dim_sizes, int dim)
{
	accessRangeA->dim = iterRange->dim = dim;
	
	accessRangeA->a1 = new int64_t[dim];
	accessRangeC->a1 = new int64_t[dim];
	
	accessRangeA->l_b = new int64_t[dim];
	accessRangeA->u_b = new int64_t[dim];
	
	accessRangeC->l_b = new int64_t[dim];
	accessRangeC->u_b = new int64_t[dim];

	iterRange->lower_bound = new int64_t[dim];
	iterRange->upper_bound = new int64_t[dim];
	
	for(int i=0; i<dim; i++)
	{
		accessRangeA->a1[i] = accessRangeC->a1[i] = 1;
		accessRangeA->l_b[i] = accessRangeC->l_b[i] = 0;
		accessRangeA->u_b[i] = accessRangeC->u_b[i] = 0;
		//iterRange->lower_bound[i] = dim_sizes[i]/2;
		iterRange->lower_bound[i] = 0;
		//iterRange->upper_bound[i] = dim_sizes[i]-1;
		iterRange->upper_bound[i] = dim_sizes[i]-1;
	}
	return;
}



