

#ifndef _DM_INTERFACE_H
#define _DM_INTERFACE_H

#include "DArray.h"
#include "DataRequest.h"
#include "pse-runtime.h"
class DArray;
class DataRequest;

class Interface {

public:
	Interface(){}
	void InsertDivisibleTask (char *_task_name, 
		pert_range_Nd_t _iterations, 
		unsigned int _num_args, 
		void **_args, 
		pert_arg_metadata_t *_args_metadata, 
		void (_host_par_func)(pert_range_Nd_t , void **), 
		void (_phi_par_func)(pert_range_Nd_t , void **, pert_arg_offset_t *, /*char *mic_1, char *mic_2, int64_t *offset_mic, unsigned int pipeline_stage,*/ pert_double_buffer_t dbuffer_metadata, char *, unsigned int mic_id),
		void (_host_join_func)(void **, void **), 
		void (_phi_map_func)(void *, void *, int64_t num_bytes),
		void (_phi_wait_func)(char *, unsigned int),
		pert_task_options_t _task_opts, 
		void **_deallocate_args, 
		void (_deallocate)(void **), 
		pert_grain_size_t *_host_grain_size, 
		pert_grain_size_t *_phi_grain_size, 
		void *_sched_hints_handle 
		);
//	void LocalInsert();
	void setArray(void *data, int i);
	void runTask(Block iterBlock);
	int distributeArray(int argNum);

private:
	char *task_name; /*name of the task */
	pert_range_Nd_t iterations;  /* iteration range with dimension infor */
	unsigned int num_args; /*number of arguments*/
	void **args; /*array of argument information*/
	pert_arg_metadata_t *args_metadata; /*array of metadata for arguments */
	void (*host_par_func)(pert_range_Nd_t , void **); /* Function pointer to Xeon host data-parallel function binary */
	void (*phi_par_func)(pert_range_Nd_t , void **, pert_arg_offset_t *, /*char *mic_1, char *mic_2, int64_t *offset_mic, unsigned int pipeline_stage,*/ 
			     pert_double_buffer_t dbuffer_metadata, char *, unsigned int mic_id);  /* Function pointer to PHI data-parallel function binary */
	void (*host_join_func)(void **, void **); /* Function pointer to Xeon host reduction function binary */
	void (*phi_map_func)(void *, void *, int64_t num_bytes); /* A map operation for the output variables coming from  phi*/
	void (*phi_wait_func)(char *, unsigned int);
	pert_task_options_t task_opts; /* Task options: priority, scheduling hints, sequential task or not, implicit completion, ... */
	void **deallocate_args; /* array of pointers to be used for deallocation */
	void (*deallocate)(void **); /* deallocation callback for reclaiming memory after task is finished */
	pert_grain_size_t *host_grain_size; /* minimum grain_size for basic tasks */
	pert_grain_size_t *phi_grain_size; /* minimum grain_size for basic tasks */
	void *sched_hints_handle;   /* handle to the LDL profiling information */
};

#endif // _DM_TASK_H
