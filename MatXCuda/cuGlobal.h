#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define CUDA_CHECK show_cuda_error(__LINE__,__FILE__);
inline void show_cuda_error(int line,char *file ){ 
	cudaThreadSynchronize();
	cudaError_t err_last=cudaGetLastError();//获得最近一次错误的错误代码
	if(err_last){
		cout<<"\nCUDA "<<cudaGetErrorString(err_last)<<" at uncertain position before line "<<line<<"  file "<<file<<"\n";//显示错误内容
		getchar();
	}
}


#define CUBLAS_CHECK(x) show_cublas_error((x),__LINE__,__FILE__);
inline void show_cublas_error(cublasStatus_t err,int line,char *file){
	if(CUBLAS_STATUS_SUCCESS!=err){
		cout<<"\nCUBLAS "<<err<<" at line "<<line<<"  file "<<file<<endl;
		getchar();
	}
}
inline void show_cublas_error(cudaError_t err,int line,char *file){
	cudaThreadSynchronize();
	if(err){
		cout<<"\nCUBLAS "<<cudaGetErrorString(err)<<"at line "<<line<<"  file "<<file<<endl;
	}
}
#define CURAND_CHECK(err)  if( (err) != CURAND_STATUS_SUCCESS ){\
        printf("CURAND error %d at file %s line %d.\n", (int)(err), __FILE__, __LINE__);\
		getchar();\
}

bool cuda_inited =false;
int cuda_threads_num;
int cuda_grid_sizeX;
int cuda_grid_sizeY;
cublasHandle_t cublasHandle;
curandGenerator_t curandGenerator;

/*
0 CUBLAS_STATUS_SUCCESS

The operation completed successfully.

1 CUBLAS_STATUS_NOT_INITIALIZED

The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup.

To correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.

2 CUBLAS_STATUS_ALLOC_FAILED

Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.

To correct: prior to the function call, deallocate previously allocated memory as much as possible.

3 CUBLAS_STATUS_INVALID_VALUE

An unsupported value or parameter was passed to the function (a negative vector size, for example).

To correct: ensure that all the parameters being passed have valid values.

4 CUBLAS_STATUS_ARCH_MISMATCH

The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.

To correct: compile and run the application on a device with appropriate compute capability, which is 1.3 for double precision.

5 CUBLAS_STATUS_MAPPING_ERROR

An access to GPU memory space failed, which is usually caused by a failure to bind a texture.

To correct: prior to the function call, unbind any previously bound textures.

6 CUBLAS_STATUS_EXECUTION_FAILED

The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.

To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.

7 CUBLAS_STATUS_INTERNAL_ERROR

An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.

To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.

8 CUBLAS_STATUS_NOT_SUPPORTED

The functionnality requested is not supported

9 CUBLAS_STATUS_LICENSE_ERROR

The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.

*/