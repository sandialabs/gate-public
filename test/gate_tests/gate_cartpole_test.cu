#include <gtest/gtest.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/cartpole/cartpole_dynamics.cuh>
#include <controllers/cartpole/cartpole_LQR.cuh>
#include <perturbations/cartpole/cartpole_perturbations.cuh>
#include <memory>
#include <chrono>
#include <iostream>


//template <class T>
//__global__ void testx0Kernel(T::all_states_at_t* x0_noise_device) {
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//	printf("Idx: %i, Mean: %f\n", idx, (*x0_noise_device).row(idx).mean());
//}

TEST(GATE_Cartpole, PropagateTrajectories) {
	//Eigen::Matrix<float, 1, 4> K;
	//K << -1., -2.55, 41.67, 11.68;

	//Eigen::Matrix<float, 4, 1> y;
	//y << -1., -2.55, 41.67, 11.68;

	//Eigen::Matrix<float, 1, 1> ans;

	//float test = (K * y)(0);
	//float test2 = K.dot(y);

	// simulation timestep
	float dt = 0.01;
	
	//auto default_guid_params = ctrl->getGuidanceParams();
	//auto default_dyn_params = ctrl->getDynamicsParams();


	//ctrl->setGuidanceParams();
	//ctrl->setDynamicsParams();
	//dyn->setParams(default_dyn_params);

	// perturbations

	CartpoleDynamics::state_array x0_mean;
	x0_mean << 01.f, 0.1f, 01.f, 0.1f;
	std::cout << "x0_mean: " << x0_mean.transpose() << std::endl;
	CartpoleDynamics::state_array x0_std;
	x0_std << 0.001f, 0.001f, 0.001f, 0.001f;
	std::cout << "x0_std: " << x0_std.transpose() << std::endl;
	CartpoleDynamics::control_array u_std;
	u_std << 0.01f;
	std::cout << "u_std: " << u_std.transpose() << std::endl;

	// Setup rollouts

	const int num_timesteps = 5000; //2000
	const int num_rollouts = 1024; //1024
	const int bdim_x = 128;

	CartpolePertParams params = CartpolePertParams(x0_mean, x0_std, u_std);

	// set params
	std::shared_ptr<CartpoleDynamics> dyn = std::make_shared<CartpoleDynamics>();
	std::shared_ptr<CartpoleLQR> ctrl = std::make_shared<CartpoleLQR>();
	std::shared_ptr<CartpolePert<num_timesteps, num_rollouts>> pert = std::make_shared<CartpolePert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();


	// Create the GATE object
	auto RTE = new GATE<CartpoleDynamics, CartpoleLQR, CartpolePert<num_timesteps, num_rollouts>, 
		num_timesteps, num_rollouts, bdim_x>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	auto start = std::chrono::system_clock::now();
	RTE->computeTrajectories();
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Total Time Elapsed: " << elapsed.count() / 1000.0 << " seconds." << std::endl;
	
	//std::cout << "Trajectory 0: " << std::endl;
	//int idx = 0;
	//float tmp;
	//int sim_idx = 0;
	//// Test will assert if all trajectories converge.
	//for (int i = 0; i < num_timesteps; ++i) {
	//	std::cout << sim_idx << "    ";
	//	for (int j = 0; j < CartpoleDynamics::STATE_DIM; ++j) {
	//		tmp = RTE->state_trajectories_host[idx];
	//		std::cout << tmp << "    ";
	//		idx++;
	//	};
	//	sim_idx++;
	//	std::cout << "" << std::endl;
	//}

	for (int i = 0; i < num_rollouts; ++i) {
		//std::cout << "Rollout: " << i << std::endl;
		//std::cout << RTE->state_trajectories_host[num_timesteps * CartpoleDynamics::STATE_DIM * i + (num_timesteps - 1) * CartpoleDynamics::STATE_DIM + 0] << ", "
		//	<< RTE->state_trajectories_host[num_timesteps * CartpoleDynamics::STATE_DIM * i + (num_timesteps - 1) * CartpoleDynamics::STATE_DIM + 1] << ", "
		//	<< RTE->state_trajectories_host[num_timesteps * CartpoleDynamics::STATE_DIM * i + (num_timesteps - 1) * CartpoleDynamics::STATE_DIM + 2] << ", "
		//	<< RTE->state_trajectories_host[num_timesteps * CartpoleDynamics::STATE_DIM * i + (num_timesteps - 1) * CartpoleDynamics::STATE_DIM + 3] << std::endl;
		for (int j = 0; j < CartpoleDynamics::STATE_DIM; ++j) {
			ASSERT_NEAR(0.0f,
				RTE->state_trajectories_host[num_timesteps * CartpoleDynamics::STATE_DIM * i // Rollout
				+ (num_timesteps - 1) * CartpoleDynamics::STATE_DIM //Timestep
				+ j], 1e-2) << "Rollout: " << i << ", State: " << j << std::endl; // State
		}
	}

	delete(RTE);


}