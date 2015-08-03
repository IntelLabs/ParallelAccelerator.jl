
#include "DataMap.h"
#include <cmath>
#include <cstring>
#include "util.h"
#include <mpi.h>

DataMap::DataMap(int _dim, int block_size, int* dim_sizes)
{
	dim = _dim;

	/* determine block size and number of blocks in each dimension */
	setBlockSize(block_size, dim_sizes);	

	dataRegion =  Block(Point(dim, dim_sizes));
	
	getMyBlocksCyclicDist(dim, myblocks, dataRegion, dataBlockSize);
}

// get send out requests based on iteration mapping
vector<DataRequest*>* DataMap::calcOutReqs(int arrayId, IterMap *im)
{
	vector<DataRequest*>* reqVec = new vector<DataRequest*>;
	
	for(int i=0; i<myblocks.size(); i++)
	{
		// find overlaps of data block
		im->findOutBlockOverlapReqs(reqVec, myblocks[i], i, arrayId);
	}
	return reqVec;
}

// find data blocks that will be received for iteration block scheduled on this PE
void DataMap::findInBlockOverlapReqs(vector<DataRequest*> *reqVec, const Block& iterBlock, int taskId, int arrayId)
{

	vector<Block> overlapBlocks;
	vector<Point> indices;
	
	dataRegion.getOverlapingSubBlocks(overlapBlocks, indices, iterBlock, dataBlockSize);
	
	for(int i=0; i<overlapBlocks.size(); i++)
	{
	       	int gblockID= blockIndexToGlobalID(indices[i]);
		int pe = globalBlockIdToPe(gblockID);
		int localBlockId = globalBlockIdToLocal(gblockID);
		reqVec->push_back(new DataRequest(arrayId, pe, taskId, localBlockId, overlapBlocks[i], IN));
	}
}

// get PE number that holds the block specified by index
int DataMap::blockIndexToGlobalID(const Point& index)
{
	int id = index[0];
	for(int i=1; i<dim; i++)
	{
		id += index[i]*(dataRegion.end[i-1]/dataBlockSize.index[i-1]);
	}
	return id;
}

// find PE that owns the block
int DataMap::globalBlockIdToPe(int gblockId)
{
	int num_pes;
	MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

	return gblockId%num_pes;
}

// find local blockId from global taskId
int DataMap::globalBlockIdToLocal(int gblockId)
{
	int num_pes;
	MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

	return gblockId/num_pes;
}

void DataMap::setBlockSize(int block_size, int* dim_sizes)
{	
	// use default block size
	if(block_size<=0) {
		if(dim==1)
		{
			int pes = 0;
			MPI_Comm_size(MPI_COMM_WORLD, &pes);
			int size = dim_sizes[0];
			block_size = size/pes;

		}
		else
			block_size = MIN_DATA_BLOCK_SIZE;
	}
	// block size in each dimension
	dataBlockSize = Point(dim, (int) pow(block_size, 1.0/dim));

	return;
}


