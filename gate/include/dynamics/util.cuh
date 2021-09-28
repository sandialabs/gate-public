#pragma once
#ifndef MPPI_DYNAMICS_UTIL_CUH
#define MPPI_DYNAMICS_UTIL_CUH

#include <iostream>
#include <math.h>
#include <math_constants.h>
#include <cuda_runtime.h>

__host__ __device__ inline float deg2rad(float deg) {
	return deg * CUDART_PI_F / 180.f;
}

__host__ __device__ inline float rad2deg(float rad) {
	return rad * 180.f / CUDART_PI_F;
}

namespace linear_alg {
	__host__ __device__ inline void vectorAdd(const float* x1, const float* x2, float* res, int size) {
		for (int i = 0; i < size; ++i) {
			res[i] = x1[i] + x2[i];
		}
	}

	__host__ __device__ inline void vectorSubtract(const float* x1, const float* x2, float* res, int size) {
		for (int i = 0; i < size; ++i) {
			res[i] = x1[i] - x2[i];
		}
	}

	__host__ __device__ inline float vectorDot(const float* x1, const float* x2, int size) {
		float result = 0.0f;
		for (int i = 0; i < size; ++i) {
			result += x1[i] * x2[i];
		}
		return result;
	}

	__host__ __device__ inline void scalarMult(const float* x1, float c, float* res, int size) {
		for (int i = 0; i < size; ++i) {
			res[i] = x1[i] * c;
		}
	}

	__host__ __device__ inline void scalarAdd(const float* x1, float c, float* res, int size) {
		for (int i = 0; i < size; ++i) {
			res[i] = x1[i] + c;
		}
	}

	__host__ __device__ inline void elementwiseMult(const float* x1, const float* x2, float* res, int size) {
		for (int i = 0; i < size; ++i) {
			res[i] = x1[i] * x2[i];
		}
	}

	__host__ __device__ inline void elementwiseAdd(const float* x1, const float* x2, float* res, int size) {
		for (int i = 0; i < size; ++i) {
			res[i] = x1[i] + x2[i];
		}
	}

	// Only defined for 3 dimensional vectors
	__host__ __device__ inline void vectorCross(const float* a,const float* b, float* d) {
		float c[3];
		c[0] = a[1] * b[2] - a[2] * b[1];
		c[1] = a[2] * b[0] - a[0] * b[2];
		c[2] = a[0] * b[1] - a[1] * b[0];

		for (int i = 0; i < 3; ++i) {
			d[i] = c[i];
		}
	}

	__host__ __device__ inline void vectorNormalize(const float* a, float* unit_a, int size) {
		// Compute the norm of a
		float norm_a = sqrtf(vectorDot(a, a, size));

		// Scalar multiply a with it 1 over its norm
		scalarMult(a, 1 / norm_a, unit_a, size);
	}

	__host__ __device__ inline float max(float a, float b) {
		return (((a) > (b)) ? (a) : (b));
	}

	__host__ __device__ inline float min(float a, float b) {
		return (((a) < (b)) ? (a) : (b));
	}

	__host__ __device__ inline float compute2x2det(const float* mat_in) {
		// Does NOT CHECK FOR SIZE!!
		// The matrix is stored as [a11 a12 a21 a22], i.e. row major;
		return (mat_in[0] * mat_in[3] - mat_in[1] * mat_in[2]);
	}

	__host__ __device__ inline float compute3x3det(const float* mat_in) {
		// Does NOT check for size
		// The matrix is stored as [a11 a12 a13 a21 a22 a23 a31 a32 a33], i.e. row major
		float submat_1[4] = { mat_in[4], mat_in[5], mat_in[7], mat_in[8] };
		float submat_2[4] = { mat_in[3], mat_in[5], mat_in[6], mat_in[8] };
		float submat_3[4] = { mat_in[3], mat_in[4], mat_in[6], mat_in[7] };
		return mat_in[0] * compute2x2det(submat_1) -
				mat_in[1] * compute2x2det(submat_2) +
				mat_in[2] * compute2x2det(submat_3);
	}

	__host__ __device__ inline void get3x3Adjoint(const float* mat_in, float* mat_out) {
		float submat_00[4] = { mat_in[4], mat_in[5], mat_in[7], mat_in[8] };
		float submat_01[4] = { mat_in[3], mat_in[5], mat_in[6], mat_in[8] };
		float submat_02[4] = { mat_in[3], mat_in[4], mat_in[6], mat_in[7] };

		float submat_10[4] = { mat_in[1], mat_in[2], mat_in[7], mat_in[8] };
		float submat_11[4] = { mat_in[0], mat_in[2], mat_in[6], mat_in[8] };
		float submat_12[4] = { mat_in[0], mat_in[1], mat_in[6], mat_in[7] };

		float submat_20[4] = { mat_in[1], mat_in[2], mat_in[4], mat_in[5] };
		float submat_21[4] = { mat_in[0], mat_in[2], mat_in[3], mat_in[5] };
		float submat_22[4] = { mat_in[0], mat_in[1], mat_in[3], mat_in[4] };
		// Take the transpose of the cofactor matrix
		mat_out[0] = compute2x2det(submat_00);
		mat_out[1] = -compute2x2det(submat_10);
		mat_out[2] = compute2x2det(submat_20);

		mat_out[3] = -compute2x2det(submat_01);
		mat_out[4] = compute2x2det(submat_11);
		mat_out[5] = -compute2x2det(submat_21);

		mat_out[6] = compute2x2det(submat_02);
		mat_out[7] = -compute2x2det(submat_12);
		mat_out[8] = compute2x2det(submat_22);
		return;
	}

	__host__ __device__ inline void invert2x2mat(const float* mat_in, float* mat_out) {
		// The matrix is stored as [a11 a12 a21 a22];

		// Compute the determinant first
		float inv_det = 1 / (mat_in[0] * mat_in[3] - mat_in[1] * mat_in[2]);

		mat_out[0] = mat_in[3] * inv_det;
		mat_out[1] = -mat_in[1] * inv_det;
		mat_out[2] = -mat_in[2] * inv_det;
		mat_out[3] = mat_in[0] * inv_det;
	}

	__host__ __device__ inline void invert3x3mat(const float* mat_in, float* mat_out) {
		get3x3Adjoint(mat_in, mat_out);
		// Can be dangerous because we are overwriting the matrix as we read
		// from it, but since this function is sequential, there should be 
		// no issues.
		scalarMult(mat_out, 1 / compute3x3det(mat_in), mat_out, 9); 
	}


	__host__ __device__ inline void matmul(const float* A, const int A_outer, const int A_inner, 
										   const float* B, const int B_outer, const int B_inner, 
												 float* C) {
		// Check to make sure that the A_inner and B_outer are equal
		if (A_inner != B_outer) {
			printf("Matrix dimensions are not compatible.");
			return;
		}

		for (int i = 0; i < A_outer; ++i) {
			for (int j = 0; j < B_inner; ++j) {
				float sum = 0;
				for (int k = 0; k < A_inner; ++k) {
					sum += A[i*A_inner + k] * B[k*B_inner + j];
				}
				C[i*B_inner + j] = sum;
			}
		}
	}

	__host__ __device__ inline void skew(const float* vec3_in, float* mat_out) {
		// Take an input 3x1 vector return the appropriate 3x3 skew symmetric matrix
		mat_out[0] = 0.f;
		mat_out[4] = 0.f;
		mat_out[8] = 0.f;

		mat_out[1] = -vec3_in[2];
		mat_out[3] = vec3_in[2];

		mat_out[2] = vec3_in[1];
		mat_out[6] = -vec3_in[1];

		mat_out[5] = -vec3_in[0];
		mat_out[7] = vec3_in[0];
	}

	__host__ __device__ inline void mat_insert(int row_index, int col_index, int mat_rows, int mat_cols, float value, float* matrix) {
		// Insert value into the matrix at given row_index and column index. Indexing starts at 0, matrix is assumed
		// to be stored in row major order.
		matrix[row_index * mat_cols + col_index] = value;
	}

	__host__ __device__ inline void solveNxN(float *aug, float *x, const int n) {
		//solves Ax = B, where A is NxN, B is Nx1
		//aug is the augmented matrix [A|B]

		//reduced echelon form
		for (int k = 0; k < n; k++) {
			for (int i = k + 1; i < n; i++) {
				/* factor f to set current row kth element to 0,
				 * and subsequently remaining kth column to 0 */
				double f = aug[i*(n+1) + k] / aug[k* (n + 1) + k];

				/* subtract fth multiple of corresponding kth
				   row element*/
				for (int j = k + 1; j <= n; j++)
					aug[i* (n + 1) + j] -= aug[k* (n + 1) + j] * f;

				/* filling lower triangular matrix with zeros*/
				aug[i* (n + 1) + k] = 0;
			}
		}

		//back substitution
		//float x[n] = { 0.0f };  // An array to store solution 

		/* Start calculating from last equation up to the
		   first */
		for (int i = n - 1; i >= 0; i--) {
			/* start with the RHS of the equation */
			x[i] = aug[i* (n + 1) + n];

			/* Initialize j to i+1 since matrix is upper
			   triangular*/
			for (int j = i + 1; j < n; j++) {
				/* subtract all the lhs values
				 * except the coefficient of the variable
				 * whose value is being calculated */
				x[i] -= aug[i* (n + 1) + j] * x[j];
			}

			/* divide the RHS by the coefficient of the
			   unknown being calculated */
			x[i] = x[i] / aug[i* (n + 1) + i];
		}
	}
	
} // namespace linear_alg

#endif // MPPI_DYNAMICS_UTIL_CUH
