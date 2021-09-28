#ifndef DUBIN_PERTURBATIONS_CUH_
#define DUBIN_PERTURBATIONS_CUH_

#include <perturbations/perturbations_stream_managed.cuh>
#include <dynamics/dubin_vehicle/dubin_dynamics.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

namespace dubin_pert {
	const int S_DIM = 3;
	const int C_DIM = 1;
}

class DubinPertParams : public GATE_internal::PerturbationParam<dubin_pert::S_DIM, dubin_pert::C_DIM> {
public:
	DubinPertParams() : GATE_internal::PerturbationParam<dubin_pert::S_DIM, dubin_pert::C_DIM>() {}
	DubinPertParams(const state_array& x0_mean, const state_array& x0_std, const control_array& u_std) :
		GATE_internal::PerturbationParam<dubin_pert::S_DIM, dubin_pert::C_DIM>(x0_mean, x0_std, u_std) {}
	~DubinPertParams() = default;
};

template<int N_TIMESTEPS, int N_ROLLOUTS>
class DubinPert : public GATE_internal::Perturbations<DubinPert<N_TIMESTEPS, N_ROLLOUTS>,
	DubinPertParams, dubin_pert::S_DIM, dubin_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS> {
public:

	DubinPert(cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<DubinPert,
		DubinPertParams, dubin_pert::S_DIM, dubin_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(stream) {
		DubinPert::allocateCUDAMem();

	}
	DubinPert(const DubinPertParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<DubinPert,
		DubinPertParams, dubin_pert::S_DIM, dubin_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(params, stream) {
		DubinPert::allocateCUDAMem();

	}
	~DubinPert() {
		DubinPert::deallocateCudaMem();
	};


	void initPerturbations() {

	}

private:
	void allocateCUDAMem() {
		
	}

	void deallocateCudaMem() {

	}
};

#endif