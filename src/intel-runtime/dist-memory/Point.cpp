#include "Point.h"
bool operator==(const Point& lhs, const Point& rhs) {
	for(int j=0; j<lhs.dim; j++)
	{
		if(lhs.index[j]!=rhs.index[j])
			return false;
	}
    return true;
}

