#include <trajectory_generation/GATE.cuh>
#include <math_constants.h>
#include <random>
#include <chrono>
#include <iostream>

namespace GATE_internal {

	template <class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
	__global__ void rolloutKernelRK4(DYN_T* dynamics_device, GUIDANCE_T* guidance_device, PERT_T* perturbations_device,
		float* state_trajectories_device,
		float dt)
	{
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_k; 
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_ktmp;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_kp1 = DYN_T::state_array::Zero();
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> xdot = DYN_T::state_array::Zero();
		Eigen::Matrix<float, DYN_T::CONTROL_DIM, 1> u_k = DYN_T::control_array::Zero(); // TODO: CANNOT CALL BLOCK FUNCTION WITHOUT HARD - CODING DIMENSIONS
		Eigen::Matrix<float, DYN_T::CONTROL_DIM, 1> u_kp1 = DYN_T::control_array::Zero();
		int tid = blockDim.x * blockIdx.x + threadIdx.x;

		// Copy the initial condition
		if (tid < NUM_ROLLOUTS) {

			x_k.block<DYN_T::STATE_DIM, 1>(0,0) << (*(perturbations_device->x0_pert_d_)).block<DYN_T::STATE_DIM, 1>(0, tid);

			// fill in states at time = 0
			for (int i = 0; i < DYN_T::STATE_DIM; ++i) {
				state_trajectories_device[tid * DYN_T::STATE_DIM * NUM_TIMESTEPS + 0 * DYN_T::STATE_DIM + i] = x_k(i);
			}

			__syncthreads();

			// Integrate dynamics forward!
			for (int k = 1; k < NUM_TIMESTEPS; ++k) {

				guidance_device->getControl(perturbations_device, k, tid, NUM_ROLLOUTS, &x_k, &xdot, &u_k, &u_kp1);

				__syncthreads();

				if (u_k.array().isNaN().sum() > 0 && tid == 13) {
					printf("WARNING:: u_k NAN detected! on timestep: %i, and rollout %i\n", k, tid);
					printf("u(%i)=%f\n", k, u_k[0]);
				}

				if (x_k.array().isNaN().sum() > 0 && tid == 13) {
					printf("WARNING:: x_k NAN detected! on timestep: %i, and rollout %i\n", k, tid);
					printf("x_k(%i)=(%f, %f)\n", k, x_k[0], x_k[1]);

				}

				integrators_eigen::rk4<DYN_T, PERT_T>(dynamics_device, perturbations_device, k, tid, x_k, u_k, dt, u_kp1, xdot, x_kp1);

				if (x_kp1.array().isNaN().sum() > 0 && tid == 13) {
					printf("WARNING:: x_kp1 NAN detected! on timestep: %i, and rollout %i\n", k, tid);
					printf("x_kp1(%i)=(%f, %f)\n", k, x_kp1[0], x_kp1[1]);
				}

				for (int i = 0; i < DYN_T::STATE_DIM; ++i) {
					x_k.block<1, 1>(i, 0) << x_kp1.block<1, 1>(i, 0);
				}

				__syncthreads();

				// Save the current state into the global variable that holds all the rollouts
				for (int i = 0; i < DYN_T::STATE_DIM; ++i) {
					state_trajectories_device[tid * DYN_T::STATE_DIM * NUM_TIMESTEPS + k * DYN_T::STATE_DIM + i] = x_k(i);
				}
			}
		}

	};
}



#define GATE_CLASS GATE<DYN_T, GUIDANCE_T, PERT_T, NUM_TIMESTEPS, NUM_ROLLOUTS, BDIM_X>

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
GATE_CLASS::GATE(DYN_T* dynamics, GUIDANCE_T* guidance, PERT_T* perturbation, state_array x0, float dt) :
	dynamics_(dynamics), guidance_(guidance), perturbation_(perturbation), dt(dt)
{

	// Set the cuda stream to the one provided by the dynamics
	stream_ = dynamics->stream_;

	// Call the GPU setup functions
	dynamics_->GPUSetup();
	guidance_->GPUSetup();
	perturbation_->GPUSetup();

	state_trajectories_host.resize(DYN_T::STATE_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS);

	// Allocate CUDA memory
	allocateCUDAMemory();
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
GATE_CLASS::~GATE()
{
	deallocateCUDAMemory();
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
void GATE_CLASS::allocateHostMemory()
{
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
void GATE_CLASS::allocateCUDAMemory()
{
	// Allocate memory for state trajectories
	HANDLE_ERROR(cudaMalloc((void**)&state_trajectories_device, DYN_T::STATE_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS * sizeof(float)));
	CudaCheckError();
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
void GATE_CLASS::deallocateCUDAMemory()
{
	cudaFree(state_trajectories_device);
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
void GATE_CLASScopyHostToDeviceArrays()
{
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
void GATE_CLASS::copyDeviceToHostArrays()
{
	cudaMemcpyAsync(state_trajectories_host.data(), state_trajectories_device, DYN_T::STATE_DIM * NUM_TIMESTEPS * NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, stream_);
	CudaCheckError();
}

template<class DYN_T, class GUIDANCE_T, class PERT_T, int NUM_TIMESTEPS, int NUM_ROLLOUTS, int BDIM_X>
void GATE_CLASS::computeTrajectories()
{
	int num_blocks = (NUM_ROLLOUTS / BDIM_X) + 1;

	// Memory is already allocated lets check how much is left
	size_t free_1, total_1, free_3, total_3;
	cudaMemGetInfo(&free_1, &total_1); // Check free/total GPU memory prior to allocation
	CudaCheckError();
	//std::cout << "Free Memory: " << free_1 << std::endl;
	//std::cout << "Total Memory: " << total_1 << std::endl;

	auto start = std::chrono::system_clock::now();
	GATE_internal::rolloutKernelRK4<DYN_T, GUIDANCE_T, PERT_T, NUM_TIMESTEPS, NUM_ROLLOUTS><<<num_blocks, BDIM_X, 0, stream_>>>(dynamics_->model_d_,
		guidance_->guidance_d_, 
		perturbation_->perturbations_d_,
		state_trajectories_device,
		dt);

	CudaCheckError();
	copyDeviceToHostArrays();
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Trajectory Compute Time: " << elapsed.count() / 1000.0 << " seconds." << std::endl;

}


#undef GATE_CLASS
