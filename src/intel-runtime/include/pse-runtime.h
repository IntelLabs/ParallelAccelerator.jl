#ifndef PSE_RUNTIME_H
#define PSE_RUNTIME_H

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* Define error codes for PSE runtime */
#define PSE_SUCCESS 0
#define ERROR -1
#define UNSUPPORTED -2

#define TRUE 1
#define FALSE 0

#define TASK_SIZE_THRESHOLD_FOR_DOUBLE_BUFFERING 4000000
#ifdef MIC_LARGE_PRE_ALLOCATION
#define PREALLOCATE_MIC_OUTPUT_BUFFER_SIZE 6000000000
#define PREALLOCATE_MIC_INPUT_BUFFER_SIZE  1900000000
#else
#define PREALLOCATE_MIC_OUTPUT_BUFFER_SIZE 1000000000
#define PREALLOCATE_MIC_INPUT_BUFFER_SIZE  1000000000
#endif

/*define error type */
typedef int error_t;

/* Task has to run synchronously */
#define TASK_FINISH (1 << 16)

/* Priority of the task */
#define TASK_PRIORITY  (1 << 15)

/* Task runs sequentially */
#define TASK_SEQUENTIAL (1 << 14)

/* Task Scheduling affinity to Xeon*/
#define TASK_AFFINITY_XEON (1 << 13)

/* Task Scheduling affinity to PHI*/
#define TASK_AFFINITY_PHI (1 << 12)

/* Task Scheduling affinity to PHI*/
#define TASK_STATIC_SCHEDULER (1 << 11)

/* use output double buffering for the task */
#define TASK_DOUBLE_BUFFER (1 << 10)

/* use input double buffering for the task */
#define TASK_INPUT_DOUBLE_BUFFER (1 << 9)


  /* Task options */
  typedef int pert_task_options_t;

  /* Argument options */
  typedef enum { 
    IN=1,          /* Read */
    OUT=2,         /* Written */
    INOUT=3,       /* Read and Written */
    VALUE=4,       /* Pass by Value and can be copied around */
    ACCUMULATOR=5  /* Allows reordering of data */
  } pert_arg_options_t;

  /* Nd iteration range */
  struct pert_range_Nd_s {
    unsigned int dim; /* Number of dimensions */
    int64_t *lower_bound; /* array of lower bounds for each dimension */
    int64_t *upper_bound; /* array of upper bounds for each dimension */
  };

  typedef struct pert_range_Nd_s pert_range_Nd_t;

  /* minimum size of basic tasks */
  struct pert_min_grain_size_Nd_s {
    unsigned int dim; /* number of dimensions */
    int64_t *min_grain_size; /* minimum blocking factor per dimension */
  };
  typedef struct pert_min_grain_size_Nd_s pert_grain_size_t;

  /* express a linear array as ai+b */
  /* Nd iteration range:  a[0] + a[1]*i0 + a[2]*i1...*/
  /* Note: for SimpleTasks: a1 and a2 arguments are not used, only l_b and u_b are used to specify the constant range for the array descriptor */
  /* we do not support a2 in our implementation */
  /* Example: Let the iteration range be [0...m]
   * Let each iteration or a subset of the iterations accesses an array A of 
   * the form A[ax+b1] to A[ax+b2], then the access descriptor assuming b2>b1 would be
   * a1[0]=a, l_b[0]=b1, u_b[0]=b2
   * if we had a 2-d array, then
   * Let each iteration or a subset of the iterations accesses an array A of 
   * the form A[a1y+b1][a2x+b2] to A[a1y+b3][a2x+b4], then the access descriptor assuming b3>b1 and b4>b2, would be
   * a1[0]=a2, l_b[0]=b2, u_b[0]=b4 -- outermost dimension is 0
   * a1[1]=a1, l_b[1]=b1, u_b[1]=b3
   */
  struct pert_array_access_desc_Nd_s {
    unsigned int dim; /* number of dimensions, if dim=0, it is a scalar value */
    int64_t *a1, *a2; /* "a" coefficients in each dimension -- a2 is not expected to be used in the current implementation */
    int64_t *l_b, *u_b; /* "b" coefficients in each dimension */
    int64_t *max_size; /* max_size in each dimension -- this might be needed for communicating data -- get rid of it later*/
    int64_t *ldims;
    bool alias;
    bool row_major; /* is the data stored row_major or column major*/
  };

  typedef struct pert_array_access_desc_Nd_s pert_array_access_desc_Nd_t; 

  /* Assumptions:
   * we only consider compressed sparse row (CSR) format for sparse matrices
   * This is described as:
   * The Intel MKL sparse matrix storage format for direct sparse solvers is
     specified by three arrays: values, columns, and rowIndex. The following table
     describes the arrays in terms of the values, row, and column positions of the
     non-zero elements in a sparse matrix.

     values: A real or complex array that contains the non-zero elements of a
      sparse matrix. The non-zero elements are mapped into the values array using the
      row-major upper triangular storage mapping described above.

     columns: Element i of the integer array columns is the number of the
      column that contains the i-th element in the values array.

     rowIndex: Element j of the integer array rowIndex gives the index of the
      element in the values array that is first non-zero element in a row j.  

     Example: The matrix B has 13 non-zero elements, and all of them are stored
      as follows:

            |  1   -1    0   -3    0  |
            | -2    5    0    0    0  |
        B = |  0    0    4    6    4  |
            | -4    0    2    7    0  |
            |  0    8    0    0   -5  |

        CSR for a Non-Symmetric Matrix with one-based indexing                             
        values  = (1  -1  -3  -2  5 4 6 4 -4  2 7 8 -5)
        columns = (1  2 4 1 2 3 4 5 1 3 4 2 5)
        rowIndex  = (1  4 6 9 12  14)      

*/
  struct pert_sparse_array_csr_access_desc_Nd_s {
    uint64_t num_rows; /* total number of rows */
    uint64_t *columns; /* as described above: Element i of the integer array columns is the number of the column that contains the i-th element in the values array.*/
    uint64_t *row_index; /* as described above: rowIndex: Element j of the integer array rowIndex gives the index of the element in the values array that is first non-zero element in a row j.  */
  };
  typedef struct pert_sparse_array_csr_access_desc_Nd_s pert_sparse_array_csr_access_desc_Nd_t; 

  struct pert_arg_metadata_s {
    pert_arg_options_t arg_opt; /* Argument options for IN, OUT, INOUT */

    bool is_scalar; /* is the argument scalar? if so, we will not transfer it to phi */
    int64_t total_size_in_bytes; /* Total size of the array in bytes -- used for bulk communication of the array*/

    int d_arr_id; /* the ID of the corresponding DArray */

    bool is_immutable; /* if this flag is set, it means the argument is assigned once and is immutable during the execution of the program. For arrays, we assume that the entire array is immutable. Use this flag with caution as runtime will do optimization of data storage and scheduling based on this flag */

    /* remaining arguments are needed by array data structures */
    unsigned int size_of_each_element_in_bytes; /*size of each element in bytes for individual array elements -- used for tracking memory locations*/
    /* is data contiguous? if not, access_desc below should provide details of data layout*/
    bool is_contiguous; /* is the data access pattern contiguous? in HPL, it is non-contiguous as at a given iteration only a subset of the data are accessed */

    pert_array_access_desc_Nd_t *array_access_desc;  /* array descriptor for the region of array accessed for a set of iterations of a divisible task*/

    bool is_sparse_array; /* if this is set, then the array is assumed to be sparse and is represented in CSR representation as described in  */
    pert_sparse_array_csr_access_desc_Nd_t *sparse_array_access_desc;  /* sparse array descriptor for the region of array accessed for a set of iteration of a divisible task*/

  };
  typedef struct pert_arg_metadata_s pert_arg_metadata_t;

  /* To handle non-contiguous data, we define argument offsets: for example, if I had a matrix 100x100 and I am accessing the matrix from 10x10 to 100x100, then we remap 10x10 to 0,0 and 100x100 to 90x90 with max_size updated to 90x90*/
  struct pert_arg_offset_s {
    unsigned int dim;
    int64_t *index_offset; /* for each dimension what is the offset to be applied to index, i.e. 10x10 is remapped to 0x0*/
    int64_t *max_size; /* for each dimension whats the max_size */
  };
  typedef struct pert_arg_offset_s pert_arg_offset_t;

  struct pert_double_buffer_s {
    bool enabled; // is double buffering of output enabled?
    char *mic_1; // Two buffers for output: mic_buffer 1
    char *mic_2; // Two buffers for output: mic_buffer 2
    int64_t *offset_mic; // offset for each output argument within the mic_1 
    char *mic_in;  // one buffer to hold the inputs
    int64_t *offset_in; // offset for each input argument in mic_in
    unsigned int pipeline_stage; // stage of the pipeline
  };
  typedef struct pert_double_buffer_s pert_double_buffer_t;

  /* Start worker threads; start the runtime */
  error_t pert_init(bool);

  /* Divisible Task insertion API -- useful for data-parallel programming
     Each function is declared in the following way:
     void host_func(pert_range_Nd_t *, void **packed_data_wrapper);
     The first thing the host_func does is to parse the arguments from packed_data_wrapper. 
     */
  error_t pert_insert_divisible_task (char *task_name, /*name of the task */
                                      pert_range_Nd_t iterations,  /* iteration range with dimension infor */
                                      unsigned int num_args, /*number of arguments*/
                                      void **args, /*array of argument information*/
                                      pert_arg_metadata_t *args_metadata, /*array of metadata for arguments */
                                      void (host_par_func)(pert_range_Nd_t , void **), /* Function pointer to Xeon host data-parallel function binary */
                                      void (phi_par_func)(pert_range_Nd_t , void **, pert_arg_offset_t *, /*char *mic_1, char *mic_2, int64_t *offset_mic, unsigned int pipeline_stage,*/ pert_double_buffer_t dbuffer_metadata, char *, unsigned int mic_id),  /* Function pointer to PHI data-parallel function binary */
                                      void (host_join_func)(void **, void **), /* Function pointer to Xeon host reduction function binary */
                                      void (phi_map_func)(void *, void *, int64_t num_bytes), /* A map operation for the output variables coming from  phi*/
                                      void (phi_wait_func)(char *, unsigned int),
                                      pert_task_options_t task_opts, /* Task options: priority, scheduling hints, sequential task or not, implicit completion, ... */
                                      void **deallocate_args, /* array of pointers to be used for deallocation */
                                      void (deallocate)(void **), /* deallocation callback for reclaiming memory after task is finished */
                                      pert_grain_size_t *host_grain_size, /* minimum grain_size for basic tasks */
                                      pert_grain_size_t *phi_grain_size, /* minimum grain_size for basic tasks */
                                      void *sched_hints_handle   /* handle to the LDL profiling information */
                                     ); 

  /* Simple Task insertion API -- useful for task-parallel programming (e.g., Fibonacci)
     Each function pointer is declared in the following way:
     void host_func(pert_range_Nd_t *, void **packed_data_wrapper);
     The first thing the host_func does is to parse the arguments from packed_data_wrapper. 
     */
  error_t pert_insert_task (char *task_name, /* name of the task */
                            unsigned int num_args, /*number of arguments*/
                            void **args, /*array of argument information*/
                            pert_arg_metadata_t *args_metadata, /*array of metadata for arguments */
                            void (host_par_func)(void **), /* Function pointer to Xeon host data-parallel function binary */
                            void (phi_par_func)(void **, unsigned int mic_id),  /* Function pointer to PHI data-parallel function binary */
                            pert_task_options_t task_opts, /* Task options: priority, scheduling hints, sequential task or not, implicit completion, ... */
                            void **deallocate_args, /* array of pointers to be used for deallocation */
                            void (deallocate)(void **), /* deallocation callback for reclaiming memory after task is finished */
                            void *sched_hints_handle
                           ); 

  /* Task Wait waits for the completion of all inserted task prior to the wait*/
  error_t pert_wait_all_task (void);


  /* Clean state on both Xeon and PHI */
  error_t pert_reset(void);

  /* Stop worker threads; stop the runtime */
  error_t pert_shutdown(void);

  /* register an array */
  error_t pert_register_data( void *data, int is_scalar, unsigned int dim, int64_t * max_size, unsigned int type_size );

  /* unregister an array */
  error_t pert_unregister_data( void *data );


#if defined(c_plusplus) || defined(__cplusplus)
}
#endif


#endif /* PSE_RUNTIME_H */
