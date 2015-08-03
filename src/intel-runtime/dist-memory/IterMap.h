
#ifndef _DM_ITERMAP_H
#define _DM_ITERMAP_H

#include <vector>
using std::vector;
#include "DataRequest.h"
#include "DataMap.h"
#include "pse-runtime.h"
#include "Task.h"

class Task;

#define MIN_ITER_BLOCK_SIZE 64

class DataMap;
class DataRequest;

// map iterations to nodes
class IterMap {
public:
	IterMap(pert_array_access_desc_Nd_t *accessRange, pert_range_Nd_t *iterRange, int iter_block_size);
	
	// find data blocks that need to be received
	vector<DataRequest*>* calcInReqs(int arrayId, DataMap *dm);
	
	// find areas of the data block that need to be sent out
	void findOutBlockOverlapReqs(vector<DataRequest*> *reqVec, const Block& dataBlock,
			int blockId, int arrId);

	static vector<Task> createLocalTasks(vector<IterMap> maps);
	int numLocalTasks() {return myIterBlocks.size();}
	Block getMyIterBlock(int i) { return myIterBlocks[i]; }
	Block getMyAccessBlock(int i) { return myAccessBlocks[i]; }
	void print()
	{
		printf("accessBlockSize:");
		accessBlockSize.print();
		printf("IterBlockSize: "); iterBlockSize.print();
		printf("accessRegion: "); accessRegion.print();
		printf("IterRegion: "); iterRegion.print();
		printf("\n");
		printf("myAccessBlocks(%lu): ", myAccessBlocks.size());
		for(int i=0; i<myAccessBlocks.size(); i++)
		{
			myAccessBlocks[i].print();
		}
		printf("\n");
	}
	
private:
	int dim;
	
	Point accessBlockSize;
	Point iterBlockSize;
	
	Block accessRegion;
	Block iterRegion;
	vector<Block> myAccessBlocks;
	vector<Block> myIterBlocks;
	
	IterMap();
	void setBlockSize(int block_size, pert_range_Nd_t *iterRange);
	
	// find global taskId from iteration block index
	int blockIndexToGlobalTaskID(const Point& index);

	// find PE that executes the task
	int globalTaskIdToPe(int taskId);
	
	// find local taskId from global taskId
	int globalTaskIdToTask(int taskId);
	// number of blocks in each dimension, used in taskId calculation blockIndexToGlobalTaskID()
	Point dimNumBlockOffsets;
	void calcDimNumBlockOffsets();
};

#endif // _DM_ITERMAP_H
