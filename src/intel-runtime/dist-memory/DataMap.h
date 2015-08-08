

#ifndef _DM_DATAMAP_H
#define _DM_DATAMAP_H

#include <vector>
using std::vector;
#include "DataRequest.h"
#include "IterMap.h"
class DataRequest;
class IterMap;

#define MIN_DATA_BLOCK_SIZE 100

// map data to nodes
class DataMap {
public:
	DataMap(int num_dims, int block_size, int* dim_sizes);
	// get send out requests based on iteration mapping
	vector<DataRequest*>* calcOutReqs(int arrId, IterMap *im);

	// find data blocks that will be received for iteration block scheduled on this PE
	void findInBlockOverlapReqs(vector<DataRequest*> *reqVec, const Block& iterBlock,
			int taskId, int arrayId);

	int getNumBlocks() { return myblocks.size(); }
	Block getBlock(int i) { return myblocks[i]; }
	// get sizes of all dimensions
	Point getDimSizes(){ return dataRegion.getDimSizes(); }
	void print()
	{
		printf("dataBlockSize: "); dataBlockSize.print();
		printf("dataRegion: "); dataRegion.print();
		printf("\n");
		printf("myblocks: ");
		for(int i=0; i<myblocks.size(); i++)
		{
			myblocks[i].print();
		}
		printf("\n");
	}
	Block dataRegion;

	int dim;
	
private:
	void setBlockSize(int block_size, int* dim_sizes);

	
	vector<Block> myblocks;
	Point dataBlockSize;

	// get PE number that holds the block specified by index
	int blockIndexToGlobalID(const Point& index);

	// find PE that owns the block
	int globalBlockIdToPe(int gblockId);

	// find local blockId from global taskId
	int globalBlockIdToLocal(int gblockId);
};

#endif // _DM_DATAMAP_H
