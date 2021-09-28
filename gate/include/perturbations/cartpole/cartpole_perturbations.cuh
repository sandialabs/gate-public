#ifndef CARTPOLE_PERTURBATIONS_CUH_
#define CARTPOLE_PERTURBATIONS_CUH_

#include <perturbations/perturbations_stream_managed.cuh>
#include <dynamics/cartpole/cartpole_dynamics.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

namespace cartpole_pert {
	const int S_DIM = 4;
	const int C_DIM = 1;
}

class CartpolePertParams : public GATE_internal::PerturbationParam<cartpole_pert::S_DIM, cartpole_pert::C_DIM> {
public:
	CartpolePertParams() : GATE_internal::PerturbationParam<cartpole_pert::S_DIM, cartpole_pert::C_DIM>() {}
	CartpolePertParams(const state_array& x0_mean, const state_array& x0_std, const control_array& u_std) :
		GATE_internal::PerturbationParam<cartpole_pert::S_DIM, cartpole_pert::C_DIM>(x0_mean, x0_std, u_std) {}
	~CartpolePertParams() = default;

	float mc_std_ = .01f; //todo
	float mp_std_ = .01f; //todo
	float  l_std_ = .01f; //todo
};


template<int N_TIMESTEPS, int N_ROLLOUTS>
class CartpolePert : public GATE_internal::Perturbations<CartpolePert<N_TIMESTEPS, N_ROLLOUTS>,
	CartpolePertParams, cartpole_pert::S_DIM, cartpole_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS> {
public:
	//CartPolePert(cudaStream_t stream = nullptr) :
	//	GATE_internal::Perturbations<CartpolePert, CartpolePertParams, cartpole_pert::S_DIM, cartpole_pert::C_DIM, cartpole_pert::N_TIMESTEPS, cartpole_pert::N_ROLLOUTS>(stream) {
	//	CartpolePert::allocateCUDAMem();
	//}
	CartpolePert(const CartpolePertParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<CartpolePert, CartpolePertParams, cartpole_pert::S_DIM, cartpole_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(params, stream) {
		CartpolePert::allocateCUDAMem();
	}
	~CartpolePert() {
		CartpolePert::deallocateCudaMem();
	};

	// Device pointers for CartpolePert derived perturbations
	scalar_all_rollouts* mc_pert_d_ = nullptr;
	scalar_all_rollouts* mp_pert_d_ = nullptr;
	scalar_all_rollouts*  l_pert_d_ = nullptr;

	void initPerturbations() {
		scalar_all_rollouts mc_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS
		scalar_all_rollouts mp_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS
		scalar_all_rollouts  l_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS

		// Scale the columns by the correct standard deviations and add the mean
		mc_pert_host = (mc_pert_host * params_.mc_std_); // TODO turn into a util?
		mp_pert_host = (mp_pert_host * params_.mp_std_);
		l_pert_host  =  (l_pert_host * params_.l_std_);


		// Transfer the data to the GPU
		HANDLE_ERROR(cudaMemcpyAsync(mc_pert_d_, &mc_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		HANDLE_ERROR(cudaMemcpyAsync(mp_pert_d_, &mp_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		HANDLE_ERROR(cudaMemcpyAsync( l_pert_d_,  &l_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));

		
		CudaCheckError();
	}

private:
	void allocateCUDAMem() {
		GATE_internal::cudaObjectMalloc(mc_pert_d_);
		GATE_internal::cudaObjectMalloc(mp_pert_d_);
		GATE_internal::cudaObjectMalloc( l_pert_d_);
	}

	void deallocateCudaMem() {
		GATE_internal::cudaObjectFree(mc_pert_d_);
		GATE_internal::cudaObjectFree(mp_pert_d_);
		GATE_internal::cudaObjectFree( l_pert_d_);
	}
};

#endif