
#include "DArray.h"
#include "util.h"

int DArray::num_arrs = 0;
vector<DArray*> DArray::dArrs;

DArray::DArray(DataMap *_map, int _elemSize): dim(_map->dim)
{
	arrID = num_arrs++;
	dArrs.push_back(this);
	
	map = _map;
	elemSize = _elemSize;

	num_blocks = map->getNumBlocks();
	data = new void*[num_blocks];
	blockSizes = new int[num_blocks];
	for(int i=0; i<num_blocks; i++)
	{
		int size = map->getBlock(i).getSize();
		blockSizes[i] = size;
		data[i] = 0; // will allocate if necessary
	}
}



vector<DataRequest*>* DArray::getOutReqs(IterMap *im) 
{
	return map->calcOutReqs(arrID, im); 
}

vector<DataRequest*>* DArray::getInReqs(IterMap *im) 
{
	return im->calcInReqs(arrID, map); 
}


void DArray::serialize(void* buffer, Block area, int blockId) 
{
	Block dataBlock = map->getBlock(blockId);
	blockAreaCopy(data[blockId], dataBlock, buffer, area, elemSize, false);
}
	
// get sizes of all dimensions
Point DArray::getDimSizes() 
{
       return map->getDimSizes(); 
}

// assemble buffers of data requests and return a local pointer
void* DArray::assembleData(vector<DataRequest*>* inReqs, Block accessBlock)
{
	if(inReqs==NULL)
		return NULL;

	if(inReqs->size()==1)
		return inReqs->at(0)->getBuffer();

	void* data = new char[accessBlock.getSize()*elemSize];

	for(int i=0; i<inReqs->size(); i++)
	{
		DataRequest* dataReq = inReqs->at(i);
		blockAreaCopy(data, accessBlock, dataReq->getBuffer(), dataReq->getArea(), elemSize, true);
	}
	return data;
}



// TODO part of a hack
// get my portions of data out of the full array
void DArray::getMyData(void* full_array)
{
	for(int i=0; i<map->getNumBlocks(); i++)
	{
		Block iBlock = map->getBlock(i);
		// Fast path 1D
		if(dim==1)
		{
			data[i] = full_array+(iBlock.start[0]*elemSize);
//			printf("array offset:%d\n",(iBlock.start[0]*elemSize));
		}
		else {
			data[i] = new char[blockSizes[i]*elemSize];
			blockAreaCopy(full_array, map->dataRegion, data[i], iBlock, elemSize, false);
		}
	}
	return;
}

DArray::~DArray()
{
	for(int i=0; i<num_blocks; i++)
		if(data[i])
			delete[] data[i];
	delete data;
	delete blockSizes;
}

// start of 1D blocks, fast path
int DArray::getBlockStart1D(int blockNo)
{
	return elemSize*(map->getBlock(blockNo).start[0]);
}


