
#ifndef _DM_DATAREQUEST_H
#define _DM_DATAREQUEST_H

#include <mpi.h>
// typedef enum {IN, OUT} ReqType;
#include "pse-runtime.h"
typedef pert_arg_options_t ReqType;

#include "Block.h"
#include "DArray.h"

// request for data to be received or sent
class DataRequest {
public:
	DataRequest(int arrayId, int _pe, int _taskId, int _blockId, const Block& _area, ReqType _req_type);
	void startComm();
	void wait();
	~DataRequest();
	void print()
	{
		printf("arrId:%d blockId:%d send:%d pe:%d taskId:%d size:%d MPI_Req:%d area:", arrayId, blockId, req_type==OUT, pe, taskId, mySize, mpi_req);
		area.print();
		printf("\n");
	}

	Block getArea() {return area;}
	void* getBuffer() {return buffer;}
	
private:
	// communication is on local processor, no MPI needed
	bool local; 
	int arrayId;
	int blockId;
	// processor number, same as MPI rank
	int pe;
	// local task id in pe
	int taskId;
	ReqType req_type;
	MPI_Request mpi_req;
	Block area;
	void* buffer;
	int mySize;
	
public:
	static void startAllComms(vector<DataRequest*>* reqs);
	static vector<DataRequest*>* getDataReqsForTask(vector<DataRequest*>* reqs, int task);
	static void waitAll(vector<DataRequest*>* reqs);
};

#endif // _DM_DATAREQUEST_H
