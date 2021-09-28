#ifndef PERTURBATIONS_STREAM_MANAGED_CUH_
#define PERTURBATIONS_STREAM_MANAGED_CUH_

/*
 * Software License Agreement (BSD License)
 * Copyright (c) 2020, Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**********************************************
 * @file perturbations_base.cu
 * @author Manan Gandhi <mgandhi@sandia.gov>
 * @date March 8, 2021
 * @copyright 2020 Georgia Institute of Technology
 * @brief Base class using CRTP to create a static
 * interface for perturbation classes.
 ***********************************************/
#include <stdio.h>

#include <cuda_util/stream_managed.cuh>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif  //_MSC_VER

#include <math.h>
#include <Eigen/Dense>
#include <random>

namespace GATE_internal {
    /*
     * Base perturbation parameter class contains at least initial condition
     * and control standard deviations. It should not be instantiated on its
     * own.
     */
    template<int S_DIM, int C_DIM>
    class PerturbationParam {
    public:
        typedef Eigen::Matrix<float, S_DIM, 1> state_array;
        typedef Eigen::Matrix<float, C_DIM, 1> control_array;

        state_array x0_mean_ = state_array::Zero();
        state_array x0_std_ = state_array::Zero();
        control_array u_std_ = control_array::Zero();
    protected:
        PerturbationParam() {}

        PerturbationParam(const state_array& x0_mean, 
            const state_array& x0_std,
            const control_array& u_std) :
            x0_mean_(x0_mean), x0_std_(x0_std), u_std_(u_std) {}

    public:
        virtual ~PerturbationParam() {};
    };

    /*
    * 	std::random_device rd;
	    std::mt19937 gen(rd());  //here you could also set a seed
	    std::normal_distribution<float> dis(0, 1);

    */
    template<class CLASS_T, class PARAMS_T, int S_DIM, int C_DIM, int NUM_TIMESTEPS, int NUM_ROLLOUTS>
    class Perturbations : public Managed {
    public:
        // Useful typedefs
        static const int STATE_DIM = S_DIM;
        static const int CONTROL_DIM = C_DIM;

        typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array; // Control at a time t
        typedef Eigen::Matrix<float, STATE_DIM, 1> state_array; // State at a time t
        typedef Eigen::Matrix<float, STATE_DIM, NUM_ROLLOUTS> all_states_at_t; // storage class for x0_perturbations
        typedef Eigen::Matrix<float, CONTROL_DIM, NUM_ROLLOUTS* NUM_TIMESTEPS> all_controls; // storage class for u_t perturbations
        typedef Eigen::Matrix<float, NUM_TIMESTEPS, NUM_ROLLOUTS> scalar_all_t_all_rollouts; // Storage class for scalar parameter for all time and rollouts
        typedef Eigen::Matrix<float, NUM_ROLLOUTS, 1> scalar_all_rollouts; // Storage class for a scalar parameters for all rollouts.

    protected:
        Perturbations(const cudaStream_t stream = nullptr) 
            : Managed(stream), 
            generator_(std::random_device{}()), // TODO figure out why this syntax is necessary
            normal_distribution_(std::normal_distribution<float>(0, 1)) {
            this->params_ = PARAMS_T();
            Perturbations::allocateCudaMem();
            //std::cout << "Constructed base perturbations." << std::endl;
        }
        Perturbations(const PARAMS_T& params, const cudaStream_t stream = nullptr)
            : Managed(stream),
            generator_(1),
            normal_distribution_(std::normal_distribution<float>(0, 1)) {
            this->params_ = params;
            Perturbations::allocateCudaMem();
            //std::cout << "Constructed base perturbations." << std::endl;
        }

        PARAMS_T params_;
        // Random number generator for creating perturbations
        std::mt19937 generator_;
        std::normal_distribution<float> normal_distribution_;

        // TODO We should make the device pointers protected and create getters for them.
        
    public:
        virtual ~Perturbations() {
            Perturbations::deallocateCudaMem();
            freeCudaMem();
            //std::cout << "Destructed base perturbations." << std::endl;

        }

        void allocateCudaMem(); // Allocates CUDA memory of perturbations
        void deallocateCudaMem();  // Deallocates CUDA memory of perturbations

        void GPUSetup(); // Allocates memory for the GPU version of this object

        void freeCudaMem(); // Deallocates memory for the GPU version of this object

        void paramsToDevice();  // Copy parameters to the GPU

        void setParams(const PARAMS_T& params) {
            params_ = params;
            if (GPUMemStatus_) {
                CLASS_T& derived = static_cast<CLASS_T&>(*this);
                derived.paramsToDevice();
            }
        }

        /*
        * Function will generate noise and copy data into the memory allocated on the GPU
        */
        void initializeX0andControlPerturbations();

        // Pointer to the GPU version of this device
        CLASS_T* perturbations_d_ = nullptr;

        /** Perturbation device pointers start here (these are common to ALL systems) **/
        all_states_at_t* x0_pert_d_ = nullptr;
        all_controls* u_t_pert_d_ = nullptr;

        __device__ __host__ PARAMS_T getParams() { return params_; }
    };

}

#ifdef __CUDACC__
#include "perturbations_stream_managed.cu"
#endif // __CUDACC__

#endif // PERTURBATIONS_STREAM_MANAGED_CUH_


/*

every dynamics has a 

computeStateDeriv(x, u, xdot, pert)

and an associated perturbation class

MSD
    mass
    spring
    damper
    kp
    ki


*/