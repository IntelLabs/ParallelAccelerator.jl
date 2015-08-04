#ifndef _DM_DARRAY_H
#define _DM_DARRAY_H

#include <vector>
using std::vector;

#include "Point.h"
#include "Block.h"
#include "DataMap.h"
#include "IterMap.h"
#include "DataRequest.h"

class DataMap;
class IterMap;

// distributed-memory array
class DArray {
public:
	DArray(DataMap *map, int elemSize);
	~DArray();
	vector<DataRequest*>* getOutReqs(IterMap *im);
	vector<DataRequest*>* getInReqs(IterMap *im);
	
	void serialize(void* buffer, Block area, int blockId); 
	
	void writeBack(vector<DataRequest*>* outReqs, void* data);
	
	int getElemSize() {return elemSize;}
	// get sizes of all dimensions
	Point getDimSizes();
	int getID(){return arrID;}
	void* getDataBlock(int blockID){return data[blockID];}
	int getBlockSize(int blockID){return blockSizes[blockID];}

	// TODO part of a hack
	// get my portions of data out of the full array
	void getMyData(void*);

	// assemble buffers of data requests and return a local pointer
	void* assembleData(vector<DataRequest*>* inReqs, Block accessBlock);
	
	static DArray* getArr(int arrId) {return dArrs[arrId];}

	int getNumBlocks(){return num_blocks;}
	
	// start of 1D blocks, fast path
	int getBlockStart1D(int blockNo);

private:

	// global id of the array
	int arrID;
	// number of dimensions
	int dim;
	// byte size of each element in the data
	int elemSize;
	// number of data blocks in local node
	unsigned num_blocks;
	// array of data blocks
	void** data;
	// map object corresponding to array
	DataMap* map;
	// size of dimensions
	Point dimSizes;
	// total number of elements in each data block (for DataRequest optimization)
	int* blockSizes;
	
	static int num_arrs;
	static vector<DArray*> dArrs;
	
};


#endif // _DM_DARRAY_H
