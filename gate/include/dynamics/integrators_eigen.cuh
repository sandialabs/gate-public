#pragma once
#ifndef NUMERICAL_INTEGRATORS_EIGEN_H
#define NUMERICAL_INTEGRATORS_EIGEN_H

#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>

// Individual Eigen includes are required to build with CUDA inside CPP kernels
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include "Eigen/geometry_wrapper.h"

namespace integrators_eigen {

	/****************************************************
	Begin function template definitions
	***************************************************/
	template< class DYN_T>
	__host__ __device__ void rk4(DYN_T* dynamics, DYN_T::state_array& x_k, DYN_T::control_array& u_k,
		float dt, DYN_T::control_array& u_kp1, DYN_T::state_array& xdot, DYN_T::state_array& x_kp1);

	template< class DYN_T, class PERT_T>
	__host__ __device__ void rk4(DYN_T* dynamics, PERT_T* pert, int timestep, int rollout, DYN_T::state_array& x_k, DYN_T::control_array& u_k,
		float dt, DYN_T::control_array& u_kp1, DYN_T::state_array& xdot, DYN_T::state_array& x_kp1);

	/****************************************************
	Begin function template implementations
	***************************************************/
	template <class DYN_T>
	__host__ __device__ void rk4(DYN_T* dynamics, DYN_T::state_array& x_k, DYN_T::control_array& u_k, 
								 float dt, DYN_T::control_array& u_kp1, DYN_T::state_array& xdot,
								 DYN_T::state_array& x_kp1) {
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k1;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k2;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k3;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k4;
		Eigen::Matrix<float, DYN_T::CONTROL_DIM, 1> u_kp_point5 = 0.5 *(u_k+u_kp1);
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_tmp;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_tmp1;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> xk_tmp;

		// compute k1
		(*dynamics).computeStateDeriv(x_k, u_k, xdot);
		k1 = dt * xdot;
		//// compute k2
		x_tmp = x_k + k1 / 2;
		(*dynamics).computeStateDeriv(x_tmp, u_kp_point5, xdot);
		k2 = dt * xdot;
		// compute k3
		x_tmp = x_k + k2 / 2; 
		(*dynamics).computeStateDeriv(x_tmp, u_kp_point5, xdot);
		k3 = dt * xdot;
		// compute k4
		x_tmp = x_k + k3;
		(*dynamics).computeStateDeriv(x_tmp, u_kp1, xdot);
		k4 = dt * xdot;
	
		x_kp1 =  x_k + (1.f / 6.f) * (k1 + 2 * k2 + 2 * k3 + k4);
	}

	// TODO UNTESTED
	template< class DYN_T, class PERT_T>
	__host__ __device__ void rk4(DYN_T* dynamics, PERT_T* pert, int timestep, int rollout, DYN_T::state_array& x_k, DYN_T::control_array& u_k,
		float dt, DYN_T::control_array& u_kp1, DYN_T::state_array& xdot, DYN_T::state_array& x_kp1) {
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k1;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k2;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k3;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> k4;
		Eigen::Matrix<float, DYN_T::CONTROL_DIM, 1> u_kp_point5 = 0.5 * (u_k + u_kp1);
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_tmp;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> x_tmp1;
		Eigen::Matrix<float, DYN_T::STATE_DIM, 1> xk_tmp;

		//PERT_T* perturbations, int timestep, int rollout,
		// compute k1
		(*dynamics).computeStateDeriv(pert, timestep, rollout, x_k, u_k, xdot);
		k1 = dt * xdot;
		//// compute k2
		x_tmp = x_k + k1 / 2;
		(*dynamics).computeStateDeriv(pert, timestep, rollout, x_tmp, u_kp_point5, xdot);
		k2 = dt * xdot;
		// compute k3
		x_tmp = x_k + k2 / 2;
		(*dynamics).computeStateDeriv(pert, timestep, rollout, x_tmp, u_kp_point5, xdot);
		k3 = dt * xdot;
		// compute k4
		x_tmp = x_k + k3;
		(*dynamics).computeStateDeriv(pert, timestep, rollout, x_tmp, u_kp1, xdot);
		k4 = dt * xdot;

		x_kp1 = x_k + (1.f / 6.f) * (k1 + 2 * k2 + 2 * k3 + k4);

	}

}

#endif // NUMERICAL_INTEGRATORS_EIGEN_H