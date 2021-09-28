#include <perturbations/perturbations_stream_managed.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

#define CLASS_Perturbation GATE_internal::Perturbations<CLASS_T, PARAMS_T, S_DIM, C_DIM, NUM_TIMESTEPS, NUM_ROLLOUTS>

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void CLASS_Perturbation::GPUSetup() {
	CLASS_T* derived = static_cast<CLASS_T*>(this);
	if (!GPUMemStatus_) {
		perturbations_d_ = Managed::GPUSetup(derived);
	}
	else {
		std::cout << "GPU Memory for Perturbation class already set." << std::endl;
	}
	derived->paramsToDevice();
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void CLASS_Perturbation::paramsToDevice() {
	if (GPUMemStatus_) {
		HANDLE_ERROR(cudaMemcpyAsync(&perturbations_d_->params_, &params_,
			sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));

		HANDLE_ERROR(cudaStreamSynchronize(stream_));
	}
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void CLASS_Perturbation::freeCudaMem() {
	if (GPUMemStatus_) {
		cudaFree(perturbations_d_);
		GPUMemStatus_ = false;
		perturbations_d_ = nullptr;
	}
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void CLASS_Perturbation::allocateCudaMem() {
	GATE_internal::cudaObjectMalloc(x0_pert_d_);
	GATE_internal::cudaObjectMalloc(u_t_pert_d_);
	//std::cout << "Allocated basePert CUDA memory" << std::endl;
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void CLASS_Perturbation::deallocateCudaMem() {
	GATE_internal::cudaObjectFree(x0_pert_d_);
	GATE_internal::cudaObjectFree(u_t_pert_d_);
	//std::cout << "Deallocated basePert CUDA memory" << std::endl;
}

template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
void CLASS_Perturbation::initializeX0andControlPerturbations() {
	// Generate noise for x0
	// Create the Gaussian noise for all states at time 0
	Eigen::MatrixXf x0_pert_host(S_DIM, NUM_ROLLOUTS);
	x0_pert_host = Eigen::MatrixXf::NullaryExpr(S_DIM, NUM_ROLLOUTS, [&]() { return normal_distribution_(generator_); });

	// Scale the columns by the correct standard deviations and add the mean
	x0_pert_host = (x0_pert_host.array().colwise() * params_.x0_std_.array()).colwise() + params_.x0_mean_.array(); // TODO turn into a util?

	x0_pert_host.col(0) = params_.x0_mean_; // Don't perturb the first initial condition
	//std::cout << "NAN CHECK x0: " << x0_pert_host.array().isNaN().sum() << std::endl;

	//std::cout << "Rowwise mean!" << std::endl;
	//std::cout << x0_pert_host.rowwise().mean() << std::endl;
	//std::cout << "Rowwise std?!" << std::endl;
	//std::cout << ((x0_pert_host.colwise() - x0_pert_host.rowwise().mean()).array().square().rowwise().sum() / (x0_pert_host.cols() - 1)).sqrt() << std::endl;
	// Transfer the data to the GPU
	HANDLE_ERROR(cudaMemcpyAsync(x0_pert_d_->data(), x0_pert_host.data(), sizeof(float) * S_DIM * NUM_ROLLOUTS, cudaMemcpyHostToDevice, stream_));
	CudaCheckError();

	// Generate noise for u
	
	// Noise is going to be CONTROL_DIM x (NUM_TIMESTEPS x NUM_ROLLOUTS)
	//Eigen::Matrix<float, C_DIM, NUM_ROLLOUTS*NUM_TIMESTEPS> u_pert_host = // This expression is fixed-size and allocated on the stack
	//	Eigen::Matrix<float, C_DIM, NUM_ROLLOUTS*NUM_TIMESTEPS >::NullaryExpr(C_DIM, NUM_ROLLOUTS*NUM_TIMESTEPS, [&]() { return normal_distribution_(generator_); });
	Eigen::MatrixXf u_pert_host(C_DIM, NUM_ROLLOUTS * NUM_TIMESTEPS);
	u_pert_host = Eigen::MatrixXf::NullaryExpr(C_DIM, NUM_ROLLOUTS * NUM_TIMESTEPS, [&]() { return normal_distribution_(generator_); });

	// Scale the columns by the correct standard deviations and add the mean
	u_pert_host = (u_pert_host.array().colwise() * params_.u_std_.array()); // TODO turn into a util?
	//std::cout << "NAN CHECK u_t: " << u_pert_host.array().isNaN().sum() << std::endl;

	// Copy to the GPU
	HANDLE_ERROR(cudaMemcpyAsync(u_t_pert_d_->data(), u_pert_host.data(), sizeof(float)* C_DIM* NUM_ROLLOUTS * NUM_TIMESTEPS, cudaMemcpyHostToDevice, stream_));
	CudaCheckError();
}