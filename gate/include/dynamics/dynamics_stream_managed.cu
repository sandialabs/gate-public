#include <dynamics/dynamics_stream_managed.cuh>

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void GATE_internal::Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::paramsToDevice() {
  if (GPUMemStatus_) {
    HANDLE_ERROR(cudaMemcpyAsync(&model_d_->params_, &params_,
      sizeof(PARAMS_T), cudaMemcpyHostToDevice,
      stream_));

    HANDLE_ERROR(cudaMemcpyAsync(&model_d_->control_rngs_,
      &control_rngs_,
      C_DIM * sizeof(float2), cudaMemcpyHostToDevice,
      stream_));

    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void GATE_internal::Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::GPUSetup() {
  CLASS_T* derived = static_cast<CLASS_T*>(this);
  if (!GPUMemStatus_) {
    model_d_ = Managed::GPUSetup(derived);
  }
  else {
    std::cout << "GPU Memory already set" << std::endl;
  }
  derived->paramsToDevice();
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM>
void GATE_internal::Dynamics<CLASS_T, PARAMS_T, S_DIM, C_DIM>::freeCudaMem() {
  if (GPUMemStatus_) {
    cudaFree(model_d_);
    GPUMemStatus_ = false;
    model_d_ = nullptr;
  }
}