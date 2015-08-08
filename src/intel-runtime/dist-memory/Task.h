

#ifndef _DM_TASK_H
#define _DM_TASK_H

#include "Interface.h"
#include "DArray.h"
#include "DataRequest.h"

class DArray;
class DataRequest;
class Interface;

class Task {

public:
	// create a local task
	Task(const Block& iterBlock, vector<DArray*> darrays, vector<Block> accessBlocks,
		       	vector<vector<DataRequest*>*> allInReqs, vector<vector<DataRequest*>*> allOutReqs,
	    Interface* interface);
	// wait for task data to be received
	void waitForData();
	// execute the task function on local node
	void execute();
	// write back output data (OUT array mode)
	void writeBack();
private:
	Interface *interface;
	Block iterBlock;
       	vector<DArray*> dArrays;
       	vector<Block> accessBlocks;
	vector<vector<DataRequest*>*> allInReqs;
	vector<vector<DataRequest*>*> allOutReqs;
	
	int num_args;
	void** myArray;

};

#endif // _DM_TASK_H
