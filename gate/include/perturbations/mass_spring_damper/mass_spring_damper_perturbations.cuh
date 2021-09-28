#ifndef MSD_PERTURBATIONS_CUH_
#define MSD_PERTURBATIONS_CUH_

#include <perturbations/perturbations_stream_managed.cuh>
#include <dynamics/mass_spring_damper/mass_spring_damper_dynamics.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

namespace msd_pert {
	const int S_DIM = 2;
	const int C_DIM = 1;
}

class MsdPertParams : public GATE_internal::PerturbationParam<msd_pert::S_DIM, msd_pert::C_DIM> {
public:
	MsdPertParams() : GATE_internal::PerturbationParam<msd_pert::S_DIM, msd_pert::C_DIM>() {}
	MsdPertParams(const state_array& x0_mean, const state_array& x0_std, const control_array& u_std) :
		GATE_internal::PerturbationParam<msd_pert::S_DIM, msd_pert::C_DIM>(x0_mean, x0_std, u_std) {}
	~MsdPertParams() = default;

	float mass_std_ = 0.05f;
};

template<int N_TIMESTEPS, int N_ROLLOUTS>
class MsdPert : public GATE_internal::Perturbations<MsdPert<N_TIMESTEPS, N_ROLLOUTS>,
	MsdPertParams, msd_pert::S_DIM, msd_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS> {
public:

	MsdPert(cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<MsdPert,
		MsdPertParams, msd_pert::S_DIM, msd_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(stream) {
		MsdPert::allocateCUDAMem();

	}
	MsdPert(const MsdPertParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<MsdPert,
		MsdPertParams, msd_pert::S_DIM, msd_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(params, stream) {
		MsdPert::allocateCUDAMem();

	}
	~MsdPert() {
		MsdPert::deallocateCudaMem();
	};

	// Device pointers for MsdPert derived perturbations
	scalar_all_rollouts* mass_pert_d_ = nullptr;

	void initPerturbations() {
		scalar_all_rollouts mass_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS

		// Scale the columns by the correct standard deviations and add the mean
		mass_pert_host = (mass_pert_host * params_.mass_std_); // TODO turn into a util?

		// Transfer the data to the GPU
		HANDLE_ERROR(cudaMemcpyAsync(mass_pert_d_, &mass_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		CudaCheckError();
	}

private:
	void allocateCUDAMem() {
		GATE_internal::cudaObjectMalloc(mass_pert_d_);
	}

	void deallocateCudaMem() {
		GATE_internal::cudaObjectFree(mass_pert_d_);
	}
};

#endif