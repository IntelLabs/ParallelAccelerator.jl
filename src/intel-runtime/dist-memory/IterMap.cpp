
#include <mpi.h>
#include "IterMap.h"
#include <cmath>
#include "util.h"

// pert interface has all inclusive intervals, but dist is left-inclusive internally. this constructor converts
IterMap::IterMap(pert_array_access_desc_Nd_t *accessRange, pert_range_Nd_t *iterRange, int block_size) 
	: dimNumBlockOffsets(iterRange->dim)
{
	dim = iterRange->dim;

	setBlockSize(block_size, iterRange);


	int *res = new int[dim]; // no need to free
	for(int i=0; i<dim; i++)
	{
		res[i] = (accessRange->u_b[i]-accessRange->l_b[i]+1) * iterBlockSize.index[i];
	}
	// access block sizes corresponding to iterations
	accessBlockSize.dim = dim; 
	accessBlockSize.index = res;
	
	// initialize access region block
	int* start = new int[dim]; // no need to free
	int* end = new int[dim]; // no need to free
	for(int i=0; i<dim; i++)
	{
		start[i] = iterRange->lower_bound[i]*accessRange->a1[i]+accessRange->l_b[i];
		// last iteration that is actually executed
		// +1 to be right-exclusive for convenience
		end[i] = iterRange->upper_bound[i]*accessRange->a1[i]+accessRange->u_b[i]+1;
	}
	accessRegion.init(dim, start, end);
	
	int* istart = new int[dim]; // no need to free
	int* iend = new int[dim]; // no need to free
	for(int i=0; i<dim; i++)
	{
		istart[i] = iterRange->lower_bound[i];
		iend[i] = iterRange->upper_bound[i]+1;
	}
	iterRegion.init(dim, istart, iend);
	calcDimNumBlockOffsets();
	
	getMyBlocksCyclicDist(dim, myAccessBlocks, accessRegion, accessBlockSize);
	getMyBlocksCyclicDist(dim, myIterBlocks, iterRegion, iterBlockSize);
	
}

// find areas of the data block that need to be sent out
void IterMap::findOutBlockOverlapReqs(vector<DataRequest*> *reqVec, const Block& dataBlock, int blockId, int arrayId)
{

	// printf("in IterMap::findOutBlockOverlapReqs, dataBlock:"); dataBlock.print(); printf("\n");
	vector<Block> overlapBlocks;
	vector<Point> indices;
	
	accessRegion.getOverlapingSubBlocks(overlapBlocks, indices, dataBlock, accessBlockSize);
	
	for(int i=0; i<overlapBlocks.size(); i++)
	{
		int globalTaskId = blockIndexToGlobalTaskID(indices[i]);
		int pe = globalTaskIdToPe(globalTaskId);
		int taskId = globalTaskIdToTask(globalTaskId);
		reqVec->push_back(new DataRequest(arrayId, pe, taskId, blockId, overlapBlocks[i], OUT));
	}
}

// find global taskId from iteration block index
int IterMap::blockIndexToGlobalTaskID(const Point& index)
{
	int taskId = index[0];
	for(int i=1; i<dim; i++)
	{
		taskId += index[i]*dimNumBlockOffsets[i];
	}
	return taskId;
}

void IterMap::calcDimNumBlockOffsets()
{
	dimNumBlockOffsets.index[0] = 1;
	for(int i=1; i<dim; i++)
	{
		int num_blocks = (accessRegion.end[i-1]-accessRegion.start[i-1])/accessBlockSize.index[i-1];
		if(num_blocks==0) 
			num_blocks = 1;
		dimNumBlockOffsets.index[i] = num_blocks*dimNumBlockOffsets[i-1];
	}
	return;
}

// find PE that executes the task
int IterMap::globalTaskIdToPe(int taskId)
{
	int num_pes;
	MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

	return taskId%num_pes;
}

// find local taskId from global taskId
int IterMap::globalTaskIdToTask(int taskId)
{
	int num_pes;
	MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

	return taskId/num_pes;
}

// find data blocks that need to be received
vector<DataRequest*>* IterMap::calcInReqs(int arrayId, DataMap *dm)
{
	vector<DataRequest*>* reqs = new vector<DataRequest*>;
	
	
	for(int i=0; i<myAccessBlocks.size(); i++)
	{
		dm->findInBlockOverlapReqs(reqs, myAccessBlocks[i], i, arrayId);
	}
	
	return reqs;
}


void IterMap::setBlockSize(int block_size, pert_range_Nd_t *iterRange)
{
// use default block size
	if(block_size<=0) {
		// simple divide for 1D arrays
		if(iterRange->dim==1)
		{
			int pes = 0;
			MPI_Comm_size(MPI_COMM_WORLD, &pes);
			int size = iterRange->upper_bound[0]+1;
			block_size = size/pes;
		}
		else
			block_size = MIN_ITER_BLOCK_SIZE;
	}
	
	// block size per dimension, replicate values
	iterBlockSize = Point(dim,(int) pow(block_size, 1.0/dim));
	return;
}	


