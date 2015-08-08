

#ifndef _DM_POINT_H
#define _DM_POINT_H

#include <cstdio>
#include <cstring>

// point in multidimensional space
class Point {
public:
	int dim;
	int *index;
	Point():dim(0), index(0) {}
//	Point(int _dim, int* _index):dim(_dim),index(_index) {}
	Point(int _dim, int* _index):dim(_dim) 
	{
		index = new int[dim];
		memcpy(index, _index, dim*sizeof(int));
	}
	
	// repeat value for all indices
	Point(int _dim, int repVal):dim(_dim) {
		index = new int[dim];
		for(int i=0; i<dim; i++)
			index[i] = repVal;
	}

	// empty point
	Point(int _dim):dim(_dim) {
		index = new int[dim];
	}

	~Point() 
	{ if(dim) delete[] index; }

	int& operator[](const int& i) const {return index[i];}

	Point( const Point& other): dim(other.dim)
	{
		index = new int[dim];
		memcpy(index, other.index, dim*sizeof(int));
	}
	Point& operator=( const Point& rhs )
	{
		if(dim) 
			delete[] index;
		dim = rhs.dim;
		index = new int[dim];
		memcpy(index, rhs.index, dim*sizeof(int));
		return *this;
	}
	void print()
	{
		
		for(int i=0; i<dim; i++)
		{
			printf("%d",index[i]);
			if(i!=dim-1)
				printf(",");
			else
				printf(" ");
		}
	}
};

bool operator==(const Point& lhs, const Point& rhs);

#endif /* _DM_POINT_H */
