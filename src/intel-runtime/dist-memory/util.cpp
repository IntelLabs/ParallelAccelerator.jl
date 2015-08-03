

#include "Point.h"
#include "Block.h"
#include <vector>
using std::vector;
#include <mpi.h>
#include <cstring>
/*
	gets a domain blocks and returns the sub blocks
	that belong to the local processor.
*/

void getMyBlocksCyclicDist(int dim, vector<Block> &myblocks, const Block& domain, const Point& blockSizes)
{

/* traverse through blocks in all dimensions 
	      and divide them between ranks in round-robin fashion */

		
	int num_pes, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// calc number of blocks in each dimension
	int* num_blocks = new int[dim];
	for(int i=0; i<dim;i++)
	{
		num_blocks[i] = (domain.end[i]-domain.start[i])/blockSizes[i];
		if(num_blocks[i]==0)
			num_blocks[i] = 1;
	}

	// indices for addressing a block
	int *indices = new int[dim];
	// next block to find
	int next = rank;
	// while there are blocks left
	while(true) {
		int left = next;
		int i=0;
		// start from 0,0..
		memset(indices, 0, dim*sizeof(int));

		while(left>0) {
			indices[i] = left%num_blocks[i];
			left /= num_blocks[i];
			i++;
			if(i==dim) break;
		}
		if(left==0) {
			int* range_starts = new int[dim];
			int* range_ends = new int[dim];
			for(int i=0; i<dim; i++) {
				range_starts[i]  =  domain.start[i] + indices[i]*blockSizes[i];
				range_ends[i]  = range_starts[i] + blockSizes[i];
				// last block in each dimension includes the leftovers
				if(indices[i]+1==num_blocks[i])
					range_ends[i] = domain.end[i];
			}
			myblocks.push_back(Block(dim, range_starts, range_ends));
			delete[] range_starts;
			delete[] range_ends;
		}
		else break;
		next += num_pes;
	}
	delete[] indices;
}


void blockAreaCopy(void* data, const Block& dataBlock, void* buffer, Block area, int elemSize, bool in)
{
	int dataOffset = 0;
	int outOffset = 0;
	
	Point dimOffsets = dataBlock.getDimSizes();
	int dim = area.dim;
	// accumulate dim sizes so they show offsets to skip for each dimension
	for(int i=1; i<area.dim; i++)
		dimOffsets.index[i] *= dimOffsets.index[i-1];
	
	// shift block to get local address from global address
	area.shiftBack(dataBlock.getStart());
	
	// local start point in data block
	Point startPoint = area.getStart();
	
	// row size of written data
	int rowSize = elemSize*area.getDimSize(0);
	
	// calculate first data offset
	dataOffset = startPoint[0];
	for(int i=1; i<area.dim; i++)
		dataOffset += startPoint[i]*dimOffsets[i-1];
	dataOffset *= elemSize;
	
	if(in)
		memcpy(data+dataOffset, buffer+outOffset, rowSize);
	else
		memcpy(buffer+outOffset, data+dataOffset, rowSize);
	// 1-D array only has one row
	if(dim==1) return;
	
	startPoint[1]++;

	while(true)
	{

		int i=1;
		while(startPoint[i]==area.end[i])
		{
			startPoint[i] = area.start[i];
			dataOffset -= (area.end[i]-area.start[i])*dimOffsets[i-1]*elemSize;
			dataOffset -= dimOffsets[0]*elemSize;
			i++;
			dataOffset += dimOffsets[i-1]*elemSize;
			if(i==dim) 
				break;
			startPoint[i]++;
		}
		if(i==dim) 
			break;
		startPoint[1]++;
		
		dataOffset += dimOffsets[0]*elemSize;
		outOffset += rowSize;
		if(in)
			memcpy(data+dataOffset, buffer+outOffset, rowSize);
		else
			memcpy(buffer+outOffset, data+dataOffset, rowSize);
	}
	
	return;
}
