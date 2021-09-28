#ifndef GATE_CUH
#define GATE_CUH

#include <dynamics/integrators_eigen.cuh>

#include <Eigen/Dense>

typedef std::vector<float> vecf;

namespace GATE_internal {

	template <class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
	__global__ void rolloutKernelRK4(DYN_T* dynamics_device, GUIDANCE_T* guidance_device, PERT_T* perturbations_device,
		float* state_trajectories_device,
		float dt);
}


template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
class GATE {
public: 
	/**
	* typedefs for access to templated class from outside classes
	**/
	typedef DYN_T TEMPLATED_DYNAMICS;
	typedef GUIDANCE_T TEMPLATED_GUIDANCE;

	int num_timesteps_ = NUM_TIMESTEPS;
	float dt;
	/**
	* Aliases
	**/
	// control typedefs
	using control_array = typename DYN_T::control_array;
	typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, NUM_TIMESTEPS> control_trajectory;
	// state typedefs
	using state_array = typename DYN_T::state_array;
	// state trajectory for one rollout
	typedef Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS> state_trajectory; 
	// state trajectory at one timestep for all rollouts
	typedef Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_ROLLOUTS> all_states_at_t;

	GATE(DYN_T* dynamics, GUIDANCE_T* guidance, PERT_T* perturbation, state_array x0, float dt);
	~GATE();

	cudaStream_t stream_; 
	DYN_T* dynamics_;
	GUIDANCE_T* guidance_;
	PERT_T* perturbation_;

	// Device total state trajectories
	float* state_trajectories_device; // Pointer to raw float array on device
	vecf state_trajectories_host; // std::vector to raw data on the host

	void computeTrajectories();


private:

	// private functions for memory allocation
	void allocateHostMemory();
	void allocateCUDAMemory();
	void deallocateCUDAMemory();

	void copyHostToDeviceArrays();
	void copyDeviceToHostArrays();

	vecf x_traj_rollouts_host;


};
#ifdef __CUDACC__
#include "GATE.cu"
#endif

#ifdef __INTELLISENSE__
#include "GATE.cu"
#endif


#endif // GATE_CUH