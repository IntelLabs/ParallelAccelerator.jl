/*
 * Special support of arguments serialization (including j2c-arrays) for pert runtime.
 * TODO: it currently only works for 64-bit, and needs some fix for 32-bit support.
 */
#ifndef J2C_ARRAY_PERT
#define J2C_ARRAY_PERT

#include <stack>

template <typename T>
class queue {
public:
	size_t first, last, max_size;
	T *data;
protected:
	bool own;
	void make_room(unsigned new_size) {
		if (max_size == 0)
		{
			max_size = new_size; 
			own = true;
			data = (T*)malloc(sizeof(T) * max_size);
		}
		if (new_size <= max_size) return;
		assert(new_size < (0x7fffffff));  // a means to restrict max size 
		while (max_size < new_size) max_size *= 2;
		if (own) {
			data = (T*)realloc(data, sizeof(T) * max_size);
		}
		else {
			T* tmp = (T*)malloc(sizeof(T) * max_size);
			memcpy(tmp, data, sizeof(T) * max_size);
			own = true;
		}
	} 
public:
	T* push_back(T &value) {
		make_room(last + 1);
		data[last] = value;
		return &data[last++];
	}
	T* pop_front() {
		assert(first < last);
		return &data[first++];   
	}
	queue() {
		first = last = max_size = 0;
		own = false;
		data = NULL;
	}
/* 
 * Caller is always responsible for managing the initialization array.   
 * When the queue is pushed, there is no guarrantee if the original array 
 * content will be modified or not.
 */
	queue(T* elems, size_t _last, size_t _max_size = 0) {
		if (!_max_size) _max_size = _last; // sort of a default value when not given
		first = 0;
		last = _last;
		max_size = _max_size;
		own = false;
		data = elems;
	}
/* 
 * After freeze, the queue is reset, and caller is responsible for managing 
 * the returned array, which may or may not point to the same array used for
 * initialization 
 */
	T* freeze(size_t *_first = NULL, size_t *_last = NULL) {
		T* tmp = data;
		if (_first) *_first = first;
		if (_last)  *_last  = last;
		first = last = max_size = 0;
		own = false;
		data = NULL;
		return tmp;
	}
	~queue() {
		if (own && data != NULL) free(data);
	}
	std::string dump() {
		std::stringstream s;
		s << "[";
		s << first << ":" << last << ":" << max_size;
		s << "](";
		for (int i = first; i < last; i++) {
			s << data[i] << ",";
		}
		s << ")";
		//std::cout << s << "\n";
		return s.str();
	}
};

typedef queue<void*>               pert_arg_queue_t;
typedef queue<pert_arg_metadata_t> pert_arg_meta_queue_t;

class pert_arg_helper : public j2c_array_io {
private:
	pert_arg_queue_t *values, *ins, *outs, *inouts, *accums;
	queue<pert_arg_metadata_t> *metas;
	std::stack<pert_arg_options_t> opts;
	pert_arg_metadata_t default_meta;

protected:
	void init() {
		default_meta.arg_opt = VALUE;
		default_meta.is_scalar = true;
		default_meta.is_immutable = false;
		default_meta.is_contiguous = true;
		default_meta.total_size_in_bytes = 0;
		default_meta.size_of_each_element_in_bytes = 0;
		default_meta.array_access_desc = NULL;
		values = ins = outs = inouts = accums = NULL;
	}

	pert_arg_queue_t *get_queue(pert_arg_options_t opt) {
		switch (opt) {
			case IN: return ins;
			case OUT: return outs;
			case INOUT: return inouts;
			case VALUE: return values;
			case ACCUMULATOR: return accums;
		}
		assert(false);
	}

	void write_value(void *value, size_t value_size, pert_arg_options_t opt) {
		pert_arg_queue_t *dest = get_queue(opt);
		if (opt == VALUE || opt == IN) { // we treat values the same for either VALUE or IN opt
			void* v = NULL;
			memcpy((void*)&v, value, value_size); // NOTE: this only works for little endian!
			dest->push_back(v);
		}
		else { // for the other opts, we directly store the location of the value
			void* v = (void*)value;
			dest->push_back(v);
		}
		if (metas) {
			pert_arg_metadata_t *m = metas->push_back(default_meta);
			if (opt == VALUE || opt == IN) m->is_scalar = true;
			else m->is_scalar = false;
			m->arg_opt = opt;
			m->total_size_in_bytes = value_size;
			m->size_of_each_element_in_bytes = value_size;
            m->is_immutable = false;
		}
	}

	void write_array(void *arr, size_t arr_length, size_t elem_size, pert_arg_options_t opt, bool immutable = false) {
		assert(arr_length > 0);
		assert(opt != VALUE);
		pert_arg_queue_t *dest = get_queue(opt);
		dest->push_back(arr);
		if (metas) {
			pert_arg_metadata_t *m = metas->push_back(default_meta);
			m->arg_opt = opt;
            m->is_immutable = immutable;
			m->is_scalar = false;
			m->is_contiguous = true;
			m->total_size_in_bytes = arr_length * elem_size;
			m->size_of_each_element_in_bytes = elem_size;
			// FIXME: By default, we assume access_desc to be 1-D only, but this is subject to review.
			// Also, we'll have memory leaks here if the array pointers in the descriptor is not
			// properly freed.
			pert_array_access_desc_Nd_t* access_desc = (pert_array_access_desc_Nd_t*)malloc(sizeof(pert_array_access_desc_Nd_t));
			m->array_access_desc = access_desc;
			access_desc->dim = 1;
			access_desc->a1 = (int64_t *)malloc(sizeof(int64_t)*1);
			access_desc->a2 = NULL;
			access_desc->max_size = (int64_t *)malloc(sizeof(int64_t)*1);
			access_desc->l_b = (int64_t *)malloc(sizeof(int64_t)*1);
			access_desc->u_b = (int64_t *)malloc(sizeof(int64_t)*1);
			access_desc->row_major = false;
			access_desc->max_size[0] = arr_length;
			access_desc->a1[0] = 1;
			access_desc->l_b[0] = 0;
			access_desc->u_b[0] = 0;
		}
	}
	// returns the pointer to the value, not the value itself
	void *read_back(size_t value_size, pert_arg_options_t opt) {
		pert_arg_queue_t *src = get_queue(opt);
		return (void*)src->pop_front();
	}

public:
/*
 * Initialization using existing main and meta queues. 
 * Note that the queue must be managed by the caller.
 */
	pert_arg_helper(pert_arg_queue_t *main, queue<pert_arg_metadata_t> *meta = NULL) {
		init();
		values = ins = outs = inouts = accums = main;
		metas = meta;
	}

/* The use of multiple queues is currently deprecated.
	pert_arg_helper(pert_arg_queue_t *_ins, pert_arg_queue_t *_outs, pert_arg_queue_t *_inouts, 
		            pert_arg_queue_t *_values, pert_arg_queue_t *_accums) {
		init();
		values = _values;
		ins    = _ins;
		outs   = _outs;
		inouts = _inouts;
		accums = _accums;
	}
*/
	void write(bool     &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(bool),     opt); }
	void write(int8_t   &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(int8_t),   opt); }
	void write(int16_t  &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(int16_t),  opt); }
	void write(int32_t  &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(int32_t),  opt); }
	void write(int64_t  &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(int64_t),  opt); }
	void write(uint8_t  &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(uint8_t),  opt); }
	void write(uint16_t &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(uint16_t), opt); }
	void write(uint32_t &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(uint32_t), opt); }
	void write(uint64_t &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(uint64_t), opt); }
	void write(float    &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(float),    opt); }
	void write(double   &x, pert_arg_options_t opt) { write_value((void*)&x, sizeof(double),   opt); }

	void write(bool     x) { write(x, VALUE); }
	void write(int8_t   x) { write(x, VALUE); } 
	void write(int16_t  x) { write(x, VALUE); } 
	void write(int32_t  x) { write(x, VALUE); } 
	void write(int64_t  x) { write(x, VALUE); } 
	void write(uint8_t  x) { write(x, VALUE); } 
	void write(uint16_t x) { write(x, VALUE); } 
	void write(uint32_t x) { write(x, VALUE); } 
	void write(uint64_t x) { write(x, VALUE); } 
	void write(float    x) { write(x, VALUE); } 
	void write(double   x) { write(x, VALUE); } 

	template <typename T>
	void write(j2c_array<T> &x, pert_arg_options_t opt, bool immutable) { 
		assert(opt != VALUE);
		opts.push(opt);
		j2c_array_copy<T>::serialize(x.num_dim, x.dims, x.data, this, immutable); 
		opts.pop();
	}
	void write_in(void *arr, uint64_t arr_length, unsigned int elem_size, bool immutable) {
		write(arr_length);
		if (arr_length == 0) return;
		write_array(arr, arr_length, elem_size, IN, immutable);
	}
	void write(void *arr, uint64_t arr_length, unsigned int elem_size, bool immutable) {
		write(arr_length);
		if (arr_length == 0) return;
		write_array(arr, arr_length, elem_size, opts.top(), immutable);
	}
/*
	void write(void *arr, uint64_t arr_length, unsigned int elem_size, pert_arg_options_t opt) {
		assert(opt != VALUE);
		opts.push(opt);
		write(arr, arr_length, elem_size);
		opts.pop();
	}
*/

	// reads with a given opt only returns the a pointer to the given value
	bool     *read(bool     x, pert_arg_options_t opt) { return (bool*)    read_back(sizeof(bool),     opt); }
	int8_t   *read(int8_t   x, pert_arg_options_t opt) { return (int8_t*)  read_back(sizeof(int8_t),   opt); }
	int16_t  *read(int16_t  x, pert_arg_options_t opt) { return (int16_t*) read_back(sizeof(int16_t),  opt); }
	int32_t  *read(int32_t  x, pert_arg_options_t opt) { return (int32_t*) read_back(sizeof(int32_t),  opt); }
	int64_t  *read(int64_t  x, pert_arg_options_t opt) { return (int64_t*) read_back(sizeof(int64_t),  opt); }
	uint8_t  *read(uint8_t  x, pert_arg_options_t opt) { return (uint8_t*) read_back(sizeof(uint8_t),  opt); }
	uint16_t *read(uint16_t x, pert_arg_options_t opt) { return (uint16_t*)read_back(sizeof(uint16_t), opt); }
	uint32_t *read(uint32_t x, pert_arg_options_t opt) { return (uint32_t*)read_back(sizeof(uint32_t), opt); }
	uint64_t *read(uint64_t x, pert_arg_options_t opt) { return (uint64_t*)read_back(sizeof(uint64_t), opt); }
	float    *read(float    x, pert_arg_options_t opt) { return (float*)   read_back(sizeof(float),    opt); }
	double   *read(double   x, pert_arg_options_t opt) { return (double*)  read_back(sizeof(double),   opt); }

	bool     read_bool()   { return *read((bool    )0, VALUE); }
	int8_t   read_int8()   { return *read((int8_t  )0, VALUE); }
	int16_t  read_int16()  { return *read((int16_t )0, VALUE); }
	int32_t  read_int32()  { return *read((int32_t )0, VALUE); }
	int64_t  read_int64()  { return *read((int64_t )0, VALUE); }
	uint8_t  read_uint8()  { return *read((uint8_t )0, VALUE); }
	uint16_t read_uint16() { return *read((uint16_t)0, VALUE); }
	uint32_t read_uint32() { return *read((uint32_t)0, VALUE); }
	uint64_t read_uint64() { return *read((uint64_t)0, VALUE); }
	float    read_float()  { return *read((float   )0, VALUE); }
	double   read_double() { return *read((double  )0, VALUE); }

	template <typename T>
	j2c_array<T> read(j2c_array<T> &x, pert_arg_options_t opt) { 
		opts.push(opt);
		x = j2c_array_copy<T>::deserialize(this); 
		opts.pop();
		return x;
	}

	void read(void **arr, uint64_t *len) {
		*len = read_uint64();
		*arr = (*len == 0) ? NULL : *(void**)read_back(*len, opts.top());
	}

	std::string dump() {
		return values->dump();
	}
};

// Assume args[0] is length, args[1] is pointer to meta array.
uintptr_t *convert_args_pointer_for_mic(void **args, unsigned MIC_DEV)
{
	uint64_t len = (uint64_t)args[0];
	uintptr_t *_args = (uintptr_t*)malloc(sizeof(uintptr_t) * len);
	pert_arg_metadata_t *meta = (pert_arg_metadata_t*)args[1];
	for (int i = 0; i < len; i++) {
		void** p = (void**)args[i];
		uintptr_t q = (uintptr_t)p;
		if (!meta[i].is_scalar) {
			switch (meta[i].arg_opt) {
			  case IN:
			  case OUT:
			  case INOUT:
// #pragma offload target(mic:MIC_DEV) in(p:length(0) alloc_if(0) free_if(0))
				{
					q = (uintptr_t)p;
				}; break;
			}
		}
		_args[i] = q;
	}
    return _args;
}

#endif
