#ifndef QUAD_PERTURBATIONS_CUH_
#define QUAD_PERTURBATIONS_CUH_

#include <perturbations/perturbations_stream_managed.cuh>
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

namespace quad_pert {
	const int S_DIM = 12;
	const int C_DIM = 6;
}

class QuadPertParams : public GATE_internal::PerturbationParam<quad_pert::S_DIM, quad_pert::C_DIM> {
public:
	QuadPertParams() : GATE_internal::PerturbationParam<quad_pert::S_DIM, quad_pert::C_DIM>() {}
	QuadPertParams(const state_array& x0_mean, const state_array& x0_std, const control_array& u_std) :
		GATE_internal::PerturbationParam<quad_pert::S_DIM, quad_pert::C_DIM>(x0_mean, x0_std, u_std) {}
	~QuadPertParams() = default;

	float m_std_ = 0.01f; //0.01f;
	float ixx_std_ = 0.25f; //.25f;
	float iyy_std_ = 0.25f; //.25f;
	float izz_std_ = 0.25f; //.25f;
};

template<int N_TIMESTEPS, int N_ROLLOUTS>
class QuadPert : public GATE_internal::Perturbations<QuadPert<N_TIMESTEPS, N_ROLLOUTS>,
	QuadPertParams, quad_pert::S_DIM, quad_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS> {
public:
	QuadPert(cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<QuadPert, QuadPertParams, quad_pert::S_DIM, quad_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(stream) {
		QuadPert::allocateCUDAMem();
	}
	QuadPert(const QuadPertParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<QuadPert, QuadPertParams, quad_pert::S_DIM, quad_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS>(params, stream) {
		QuadPert::allocateCUDAMem();
	}
	~QuadPert() {
		QuadPert::deallocateCudaMem();
	};

	// Device pointers for QuadPert derived perturbations
	scalar_all_rollouts* m_pert_d_ = nullptr;
	scalar_all_rollouts* ixx_pert_d_ = nullptr;
	scalar_all_rollouts* iyy_pert_d_ = nullptr;
	scalar_all_rollouts* izz_pert_d_ = nullptr;
	//scalar_all_rollouts*  l_pert_d_ = nullptr;

	void initPerturbations() {
		scalar_all_rollouts m_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS
		scalar_all_rollouts ixx_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS
		scalar_all_rollouts iyy_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS
		scalar_all_rollouts izz_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS

		//scalar_all_rollouts  l_pert_host = scalar_all_rollouts::NullaryExpr(1, quad_pert::N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS

		// Scale the columns by the correct standard deviations and add the mean
		m_pert_host = (m_pert_host * params_.m_std_); // TODO turn into a util?
		ixx_pert_host = (ixx_pert_host * params_.ixx_std_);
		iyy_pert_host = (iyy_pert_host * params_.iyy_std_);
		izz_pert_host = (izz_pert_host * params_.izz_std_);


		// Transfer the data to the GPU
		HANDLE_ERROR(cudaMemcpyAsync(m_pert_d_, &m_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		HANDLE_ERROR(cudaMemcpyAsync(ixx_pert_d_, &ixx_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		HANDLE_ERROR(cudaMemcpyAsync(iyy_pert_d_, &iyy_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		HANDLE_ERROR(cudaMemcpyAsync(izz_pert_d_, &izz_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		//HANDLE_ERROR(cudaMemcpyAsync( l_pert_d_,  &l_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));

		
		CudaCheckError();
	}

private:
	void allocateCUDAMem() {
		GATE_internal::cudaObjectMalloc(m_pert_d_);
		GATE_internal::cudaObjectMalloc(ixx_pert_d_);
		GATE_internal::cudaObjectMalloc(iyy_pert_d_);
		GATE_internal::cudaObjectMalloc(izz_pert_d_);
		//GATE_internal::cudaObjectMalloc( l_pert_d_);
	}

	void deallocateCudaMem() {
		GATE_internal::cudaObjectFree(m_pert_d_);
		GATE_internal::cudaObjectFree(ixx_pert_d_);
		GATE_internal::cudaObjectFree(iyy_pert_d_);
		GATE_internal::cudaObjectFree(izz_pert_d_);
		//GATE_internal::cudaObjectFree( l_pert_d_);
	}
};

#endif