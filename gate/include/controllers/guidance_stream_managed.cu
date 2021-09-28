#include <controllers/guidance_stream_managed.cuh>

/*
 * This cuda source file holds the implementation of functions that utilize CUDA API specific functions, such as
 * memcopies, cudaFree, and allocation. This is to help the compiler when creating shared libraries, only the
 * files that require nvcc will utilize these implementations, otherwise all other files will have the header
 * without the cuda specific api functions.
 */

template<class CLASS_T, class PARAMS_T, class DYN_PARAMS_T, int S_DIM, int C_DIM>
void GATE_internal::Guidance<CLASS_T, PARAMS_T, DYN_PARAMS_T, S_DIM, C_DIM>::GPUSetup() {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    if (!GPUMemStatus_) {
        guidance_d_ = Managed::GPUSetup(derived);
    }
    else {
        std::cout << "GPU Memory already set" << std::endl;
    }
    derived->paramsToDevice();
}

template<class CLASS_T, class PARAMS_T, class DYN_PARAMS_T, int S_DIM, int C_DIM>
void GATE_internal::Guidance<CLASS_T, PARAMS_T, DYN_PARAMS_T, S_DIM, C_DIM>::paramsToDevice() {
    if (GPUMemStatus_) {
        HANDLE_ERROR(cudaMemcpyAsync(&guidance_d_->guidance_params_, &guidance_params_,
            sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));

        HANDLE_ERROR(cudaMemcpyAsync(&guidance_d_->dynamics_params_, &dynamics_params_,
            sizeof(DYN_PARAMS_T), cudaMemcpyHostToDevice, stream_));

        HANDLE_ERROR(cudaStreamSynchronize(stream_));
    }
}

template<class CLASS_T, class PARAMS_T, class DYN_PARAMS_T, int S_DIM, int C_DIM>
void GATE_internal::Guidance<CLASS_T, PARAMS_T, DYN_PARAMS_T, S_DIM, C_DIM>::freeCudaMem() {
    if (GPUMemStatus_) {
        cudaFree(guidance_d_);
        GPUMemStatus_ = false;
        guidance_d_ = nullptr;
    }
}