
#include "Task.h"


// create a local task
Task::Task(const Block& _iterBlock, vector<DArray*> _darrays, vector<Block> _accessBlocks,
		vector<vector<DataRequest*>*> _allInReqs, 
		vector<vector<DataRequest*>*> _allOutReqs,
		Interface* _interface): iterBlock(_iterBlock),
							dArrays(_darrays),
							accessBlocks(_accessBlocks),
							allInReqs(_allInReqs),
							allOutReqs(_allOutReqs),
							interface(_interface)
{
	num_args = _darrays.size();
	myArray = new void*[num_args];
}

// wait for task data to be received
void Task::waitForData()
{
	for(int i=0; i<num_args; i++)
	{
		vector<DataRequest*>* arrReqs = allInReqs[i];
		DataRequest::waitAll(arrReqs);
		myArray[i] = dArrays[i]->assembleData(arrReqs, accessBlocks[i]);
		interface->setArray( myArray[i], i);
	}
}

// TODO
// execute the task function on local node
void Task::execute()
{
	interface->runTask(iterBlock);
}

// TODO
// write back output data (OUT array mode)
void Task::writeBack()
{
}
