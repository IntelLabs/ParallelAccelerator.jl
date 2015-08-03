
#include <mpi.h>
#include "DataRequest.h"

int calcTag(int arrID, Block area);

DataRequest::DataRequest(int _arrayId, int _pe, int _taskId, int _blockId, const Block& _area, ReqType _req_type)
		:arrayId(_arrayId), pe(_pe), taskId(_taskId), blockId(_blockId), area(_area), req_type(_req_type), mySize(0)
{
	local = false;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// if data is locally available
	if(pe==rank)
	{
		local = true;
		DArray* A = DArray::getArr(arrayId);
		// pass pointer if receiving the whole block locally
		if(req_type==IN && area.getSize()==A->getBlockSize(blockId))
		{
			buffer = A->getDataBlock(blockId);
		}
		return;
	}
	int tag = calcTag(arrayId, area);
	int size = area.getSize();
	int dataElemSize = DArray::getArr(arrayId)->getElemSize();
	buffer = new char[dataElemSize*size];

	// save size for debugging
	mySize = dataElemSize*size;
	
	if(req_type==OUT)
		MPI_Send_init(buffer, dataElemSize*size, MPI_CHAR, pe, tag, MPI_COMM_WORLD, &mpi_req);
	else
		MPI_Recv_init(buffer, dataElemSize*size, MPI_CHAR, pe, tag, MPI_COMM_WORLD, &mpi_req);
		
}

int calcTag(int arrID, Block area)
{
	int serialAddr = area.start[0];
	
	DArray *A = DArray::getArr(arrID);
	Point Size = A->getDimSizes();
	
	for(int i=1; i<area.dim; i++)
	{
		serialAddr += Size.index[i-1]*area.start[i];
	}
	
	return serialAddr+(arrID<<20);
}


void DataRequest::startComm()
{
	if(local)
		return;
	if(req_type==OUT)
	{
		DArray *A = DArray::getArr(arrayId);
		A->serialize(buffer, area, blockId);
	}
	MPI_Start(&mpi_req);
}

void DataRequest::wait()
{
	if(!local)
		MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
}


void DataRequest::startAllComms(vector<DataRequest*>* reqs)
{
	for(int i=0; i<reqs->size(); i++)
	{
		reqs->at(i)->startComm();
	}
}

vector<DataRequest*>* DataRequest::getDataReqsForTask(vector<DataRequest*>* reqs, int task)
{
	vector<DataRequest*>* out = new vector<DataRequest*>;
	
	for(int i=0; i<reqs->size(); i++)
	{
		if(reqs->at(i)->taskId ==task)
			out->push_back(reqs->at(i));
	}
	return out;
}

void DataRequest::waitAll(vector<DataRequest*>* reqs)
{
	if(reqs==NULL) return;
	for(int i=0; i<reqs->size(); i++)
	{
		reqs->at(i)->wait();
	}
}

DataRequest::~DataRequest()
{
	delete[] buffer;
	if(!local)
		MPI_Request_free(&mpi_req);
}
