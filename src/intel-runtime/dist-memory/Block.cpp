#include "Block.h"
#include <cstring>

Block::Block(int _dim, int* _start, int* _end): dim(_dim)
{

		start = new int[dim];
		end = new int[dim];
		memcpy(start, _start, dim*sizeof(int));
		memcpy(end, _end, dim*sizeof(int));
}

Block& Block::operator=( const Block& rhs )
{
	if(!dim)
	{
		dim = rhs.dim;
		start = new int[dim];
		end = new int[dim];
	}
	memcpy(start, rhs.start, dim*sizeof(int));
	memcpy(end, rhs.end, dim*sizeof(int));
	return *this;
}

// all bounds divided by point values in each dimension
Block Block::operator/(const Point &b) const
{
	int* _start = new int[dim]; // no need to free
	int* _end = new int[dim]; // no need to free
	for(int i=0; i<dim; i++)
	{
		_start[i] = start[i]/b[i];
		_end[i] = end[i]/b[i];
	}
	Block res;
	res.init(dim, _start, _end);
	return res; 
}

// copy constructor
Block::Block(const Block& other): dim(other.dim)
{
	start = new int[dim];
	end = new int[dim];
	memcpy(start, other.start, dim*sizeof(int));
	memcpy(end, other.end, dim*sizeof(int));
}

// get sizes of all dimensions
Point Block::getDimSizes() const
{
	int* index = new int[dim];
	for(int i=0; i<dim; i++)
		index[i] = end[i]-start[i];
	return Point(dim, index);
}

// get starting point of the block
Point Block::getStart() const
{
	int* index = new int[dim];
	for(int i=0; i<dim; i++) 
		index[i] = start[i];
	return Point(dim, index);
}

// get ending point of the block
Point Block::getEnd() const
{
	int* index = new int[dim];
	for(int i=0; i<dim; i++) 
		index[i] = end[i];
	return Point(dim, index);
}

// find sub-blocks of this block that overlap with the input block
void Block::getOverlapingSubBlocks(vector<Block> &overlapBlocks, vector<Point> &block_indices,
		const Block& dataBlock, const Point& blockSize)
{	
	/* scalable algorithm: find the overall overlap and only search through possible sub block overlaps */

	// get overall overlap
	Block overlap = this->getOverlap(dataBlock);

	// no overlap if empty
	if(overlap.isEmpty())
		return;

	Block indexRange = this->findSubblockIndRange(overlap, blockSize);


	// start of index range
	Point startInd = indexRange.getStart();

	// end of index range
	Point endInd = indexRange.getEnd();

	Point subBlockInd = startInd;

	while(true)
	{	
		// get sub block overlap
		Block subBlock = getBlockFromInd(subBlockInd, blockSize);
		Block area = subBlock.getOverlap(overlap);
		
		// no overlap if empty
		if(!area.isEmpty())
		{
			// save overlap area
			overlapBlocks.push_back(area);
			// sub block index
			block_indices.push_back(subBlockInd);
		}
		int i=0;
		subBlockInd[0]++;
		while(subBlockInd[i]==endInd[i])
		{
			subBlockInd[i] = startInd[i];
			i++;
			if(i==dim) break;
			subBlockInd[i]++;
		}
		if(i==dim)
			break;
	}

	return;	
}

Block Block::findSubblockIndRange(const Block& overlap, const Point& blockSize)
{
	Block indexRange(dim);
       for(int i=0; i<dim; i++)
       {
	       int num_blocks = (end[i]-start[i])/blockSize[i];
	       if(num_blocks==0) 
		       num_blocks=1;
	       indexRange.start[i] = (overlap.start[i]-this->start[i])/blockSize[i];
	       // corner case if start in last block's remainder
	       if(indexRange.start[i]==num_blocks)
		       indexRange.start[i]--;
	       indexRange.end[i] = (overlap.end[i]-this->start[i])/blockSize[i];
	       indexRange.end[i]++; // make it exclsive
	       if(indexRange.end[i]>num_blocks)
		       indexRange.end[i]=num_blocks;

       }
	return indexRange;       
}
/*
void Block::adjustRange(Block& indexRange, Point& blockSize)
{
	for(int i=0; i<dim; i++)
	{
		int num_blocks = (end[i]-start[i])/blockSize[i];
		if(indexRange.start[i]==num_blocks)
			indexRange.start[i]--;
		if(indexRange.end[i]!=num_blocks)
			indexRange.end[i]++;
	}
}
*/

// get sub block from index and block size
Block Block::getBlockFromInd(const Point& subBlockInd, const Point& blockSize)
{
	int* substart = new int[dim];
	int* subend = new int[dim];

	for(int i=0; i<dim; i++)
	{
		int num_blocks = (end[i]-start[i])/blockSize[i];
		substart[i] = start[i] + subBlockInd[i]*blockSize[i];
		subend[i] =  start[i] + (subBlockInd[i]+1)*blockSize[i];
		if(subBlockInd[i]+1 == num_blocks)
			subend[i] = end[i];
	}
	return Block(dim, substart, subend);
}


// get the corner specified by a bitmap
Point Block::getCorner(int b)
{
	// indices of corner point
		int* point = new int[dim];
		// set corner indices
		for(int i=0; i<dim; i++)
		{
			if(b&(1<<i)) {
				point[i] = start[i];
			}
			else 
			{
				point[i] = end[i];
			}
		}
	return Point(dim, point);
}

// find the index of sub-block that includes the point
Point Block::findSubBlockIndex(const Point& point, const Point& accessBlockSize)
{

	// find index of access block if point is inside access region
	bool valid = true;
	int* block_index = new int[dim];
	for(int i=0; i<dim; i++)
	{
		if(point[i]<this->start[i] || point[i]>=this->end[i])
		{
			block_index[0] = -1;
			break;
		}
		block_index[i] = (point[i]-start[i])/accessBlockSize.index[i];
	}
	return Point(dim, block_index);
}

// find overlapping area of this block with the input block
Block Block::getOverlap(const Block& dataBlock)
{
	int *bstart = new int[dim];
	int *bend = new int[dim];
	
	for(int i=0; i<dim; i++)
	{
		bstart[i] = MAX(dataBlock.start[i], start[i]);
		bend[i] = MIN(dataBlock.end[i], end[i]);
	}
	return Block(dim, bstart, bend);
}

// get the total size of this block
int Block::getSize()
{
	int size=1;
	for(int i=0; i<dim; i++)
		size *= (end[i]-start[i]);
	return size;
}


// initialize 
void Block::init(int _dim, int* _start, int* _end)
{
	dim = _dim;

	start = _start;
	end = _end;
}


// create block starting from (0,0,0..) to ends
Block::Block(Point ends) : dim(ends.dim)
{
	start = new int[dim];
	end = new int[dim];
	memset(start, 0, dim*sizeof(int));
	memcpy(end, ends.index, dim*sizeof(int));
}

// shift block back in multidimensional space 
void Block::shiftBack(const Point& p)
{
	for(int i=0; i<dim; i++)
	{
		start[i] -= p[i];
		end[i] -= p[i];
	}
}

bool Block::isEmpty() const
{
	for(int i=0; i<dim; i++)
		if(start[i]>=end[i])
			return true;
	return false;
}


pert_range_Nd_t Block::toRangeT()
{
	pert_range_Nd_t range;
	range.dim=dim;
	range.lower_bound = new int64_t[dim];
	range.upper_bound = new int64_t[dim];
	for(int i=0; i<dim; i++)
	{
		range.lower_bound[i] = start[i]; 
		range.upper_bound[i] = end[i]-1; 
	}
	return range;
}
