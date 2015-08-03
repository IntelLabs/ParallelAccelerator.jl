
#ifndef _DM_UTIL_H
#define _DM_UTIL_H


void getMyBlocksCyclicDist(int dim, vector<Block> &myblocks, const Block& domain, const Point& blockSizes);

void blockAreaCopy(void* data, const Block& dataBlock, void* buffer, Block area, int elemSize, bool in);

#endif // _DM_UTIL_H
