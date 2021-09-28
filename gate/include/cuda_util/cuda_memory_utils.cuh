#ifndef CUDA_MEMORY_UTILS_CUH_
#define CUDA_MEMORY_UTILS_CUH_

/* 
 * Header containing useful functions for manipulating CUDA memory.
 *
 */
#include <gpu_err_chk.h>
#include <new>

namespace GATE_internal {
	template<class T>
	void cudaObjectMalloc(T*& ptr_d_, int num_objects=1) {
		if (ptr_d_ == nullptr) {
			HANDLE_ERROR(cudaMalloc((void**)&ptr_d_, sizeof(T)*num_objects));
			CudaCheckError();
		}
		else {
			fprintf(stderr, "Attempted to allocate CUDA memory to occupied location!");
			throw std::bad_alloc();
		}
	}

	template<class T>
	void cudaObjectFree(T*& ptr_d_) {
		if (ptr_d_ != nullptr) {
			HANDLE_ERROR(cudaFree(ptr_d_));
			ptr_d_ = nullptr;
			CudaCheckError();
		} else {
			fprintf(stderr, "Attempted to deallocate CUDA memory from undefined location!");
			throw std::bad_alloc();
		}
	}

	template<class T>
	void cudaMatrixPertGenAndMemcpy() {

	}

	template<class T>
	void cudaTensorPertGenAndMemcpy() {

	}

}
#endif CUDA_MEMORY_UTILS_CUH_
