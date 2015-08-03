
#ifndef _DM_BLOCK_H
#define _DM_BLOCK_H

#include "Point.h"
#include <vector>
using std::vector;
#include "pse-runtime.h"

#include <cstdio>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// block in multidimensional space
class Block {
public:
	int dim;
	// start of global address of block
	int* start;
	// end of global address of block
	int* end;

	Block():dim(0), start(0), end(0) {}
	Block(int _dim):dim(_dim) {start=new int[dim]; end=new int[dim];}

	// get the total size of this block
	int getSize();

	int getDimSize(int i) const { return end[i]-start[i]; }

	// initialize block, pointers not copied! 
	void init(int dim, int* start, int* end);

	Block(int _dim, int* _start, int* _end);
	// copy constructor	
	Block( const Block& other);
	Block& operator= (const Block& rhs );

	// all bounds divided by point values in each dimension
	Block operator/(const Point &b) const;

	// create block starting from (0,0,0..) to ends
	Block(Point ends);

	// get sizes of all dimensions
	Point getDimSizes() const;

	// get starting point of the block
	Point getStart() const;
	
	// get ending point of the block
	Point getEnd() const;

	pert_range_Nd_t toRangeT();
	
	// shift block back in multidimensional space 
	void shiftBack(const Point& p);
	
	// find sub-blocks of this block that overlap with the input block
	void getOverlapingSubBlocks(vector<Block> &overlapBlocks, 
		vector<Point> &block_indices, const Block& dataBlock, const Point& accessBlockSize);

	// get the corner specified by a bitmap
	Point getCorner(int b);

	// find the index of sub-block that includes the point
	Point findSubBlockIndex(const Point& point, const Point& accessBlockSize);

	// find overlapping area of this block with the input block
	Block getOverlap(const Block& dataBlock);

	// get sub block from index and block size
	Block getBlockFromInd(const Point& subBlockInd, const Point& blockSize);
	Block findSubblockIndRange(const Block& overlap, const Point& blockSize);

	bool isEmpty() const;

	void print()
	{
		for(int i=0; i<dim; i++)
		{
			printf("%d:%d",start[i],end[i]);
			if(i!=dim-1)
				printf(",");
			else
				printf(" ");
		}
	}
	
	~Block()
	{
		if(dim) {
		delete[] start;
		delete[] end;
		}
	}
};


#endif /* _DM_BLOCK_H */
