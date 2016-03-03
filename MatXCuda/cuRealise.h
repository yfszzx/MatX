namespace gpu_funcs{
	template <typename TYPE>
	struct sigm {  
		__device__ TYPE operator ()( TYPE & x) const {  
			return 1.0f/(1.0f+__expf(-x));
		}  
	};
	template <typename TYPE>
	struct sigmDeriv {  
		__device__ TYPE operator ()( TYPE & diff, TYPE & val) const {  
			return diff * val * (1.0f - val);
		}  
	};

	template <typename TYPE>
	struct tanh {  
		__device__ TYPE operator ()( TYPE & x) const {  
			return 1.0f-2.0f/(1.0f+__expf(2*x));
		}  
	};
	template <typename TYPE>
	struct tanhDeriv {  
		__device__ TYPE operator ()( TYPE & diff, TYPE & val) const {  
			return diff * (1.0f - val * val);
		}  
	};
	template <typename TYPE>
	struct square {  
		__device__ TYPE operator ()( TYPE & x) const {  
			return x*x; 
		}  
	};
	template <typename TYPE>
	struct product {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return x*y; 
		}  
	};
	template <typename TYPE>
	struct quotient {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return x/y; 
		}  
	};
	template <typename TYPE>
	struct cwiseInverse {  
		__device__ TYPE operator ()(const  TYPE & x) const {  
			return 1/x; 
		}  
	};
	template <typename TYPE>
	struct absf {  
		__device__ TYPE operator ()(const  TYPE & x) const {  
			return (x>0)?x:(-x); 
		}  
	};
	template <typename TYPE>
	struct sqrt {  
		__device__ TYPE operator ()(const  TYPE & x) const {  
			return ::sqrt(x); 
		}  
	};
	template <typename TYPE>
	struct GreEqu {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return (x >= y)?1:0;
		}  
	};
	template <typename TYPE>
	struct Equ {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return (x == y)?1:0;
		}  
	};
	template <typename TYPE>
	struct LesEqu {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return (x <= y)?1:0;
		}  
	};
	template <typename TYPE>
	struct Les {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return (x < y)?1:0;
		}  
	};
	template <typename TYPE>
	struct Gre {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return (x > y)?1:0;
		}  
	};
	template <typename TYPE>
	struct NotEqu {  
		__device__ TYPE operator ()(const  TYPE & x, const TYPE & y) const {  
			return (x != y)?1:0;
		}  
	};
	struct flt2dbl{  
		__device__ double operator ()(const float & x) const {  
			return double(x);
		}  
	};
	struct dbl2flt {  
		__device__ float operator ()(const double & x) const {  
			return float(x);
		}  
	};
	struct plusFloatMat {  
		const double sclD;
		const float sclS;
		plusFloatMat(double D, float S):sclD(D),sclS(S){};
		__host__ __device__ double operator ()(const double & dest, const float & src ) const {  
			return dest * sclD + (double)src *sclS;
		}  
	};
}

namespace cuda_params{
	float onef = 1;
	float zerof = 0;
	double oned = 1;
	double zerod = 0;
	float *one_array_f ;
	double *one_array_d ;
	int one_len_f;
	int one_len_d;
	void init_one_array(){
		one_len_f = 1000;
		one_len_d = 1000;
		cuWrap::malloc((void **)&one_array_f, sizeof(float) * one_len_f);
		cuWrap::malloc((void **)&one_array_d, sizeof(double) * one_len_d);
		thrust :: device_ptr <float> f (one_array_f );
		thrust::fill(f, f + one_len_f , 1);
		thrust :: device_ptr <double> d (one_array_d );
		thrust::fill(d, d + one_len_d , 1);

	}
	inline void change_one_len_f(int len){
		if(len >  one_len_f){
			one_len_f = len;
			cuWrap::free(one_array_f);
			cuWrap::malloc((void **)&one_array_f, sizeof(float) * one_len_f);
			thrust :: device_ptr <float> f (one_array_f );
			thrust::fill(f, f + one_len_f , 1);
		}
	}
	inline void change_one_len_d(int len){
		if(len >  one_len_d){
			len = one_len_d;
			cuWrap::free(one_array_d);
			cuWrap::malloc((void **)&one_array_d, sizeof(double) * one_len_d);
			thrust :: device_ptr <double> d (one_array_d );
			thrust::fill(d, d + one_len_d , 1);
		}
	}
};
void cuWrap::initCuda(bool showGpuInfo){
	if(cuda_inited)return;
	int device_count;
	if( cudaGetDeviceCount(&device_count) ){
		cout<<"\n【错误】没有发现可用的显卡设备";
		getchar();
		return;
	}
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cuda_threads_num = prop.maxThreadsPerBlock;
	cuda_grid_sizeX = prop.maxGridSize[0];
	cuda_grid_sizeY = prop.maxGridSize[1];
	CUBLAS_CHECK(cublasCreate(&cublasHandle));
	cuda_inited = true;
	
	cuda_params::init_one_array();
	if(showGpuInfo){
		cout<<"\nGPU:"<<prop.name;
		/*
		cout<<"\ncuda_threads_num:"<<cuda_threads_num;
		cout<<"\ncuda_grid_sizeX:"<<cuda_grid_sizeX;
		cout<<"\ncuda_grid_sizeY:"<<cuda_grid_sizeY;
		*/
	}
};
void cuWrap::malloc(void ** p,size_t s){
	cudaMalloc(p,s);	
	CUDA_CHECK;
};
void cuWrap::free(void * p){
	cudaFree(p);
	CUDA_CHECK;
};
void cuWrap::fill(float *p, float value, int len){
	thrust :: device_ptr <float> v( p); 
	thrust::fill(v, v + len, value);
};
void cuWrap::fill(double *p, double value, int len){
	thrust :: device_ptr <double > v( p ); 
	thrust::fill(v, v + len, value);
};
void cuWrap::memD2H(void *to, const void *from, size_t size){
	cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
	CUDA_CHECK;
};
void cuWrap::memH2D(void *to, const void *from, size_t size){
	cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
	CUDA_CHECK;
};
void cuWrap::memD2D(void *to, const void *from, size_t size){
	cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice);
	CUDA_CHECK;
};
void cuWrap::memDf2Hd(double  *dest, const float  *src, int size){
	double * tmp;
	cuWrap::malloc((void **)& tmp, sizeof(double) * size);
	cuWrap::memDf2Dd(tmp, src, size);
	cuWrap::memD2H(dest, tmp, sizeof(double) * size);
	cuWrap::free(tmp);
};
void cuWrap::memHf2Dd(double *dest, const float *src, int size){
	float * tmp;
	cuWrap::malloc((void **)& tmp, sizeof(float) * size);
	cuWrap::memH2D(tmp, src, sizeof(float) * size);
	cuWrap::memDf2Dd(dest, tmp, size);	
	cuWrap::free(tmp);

};
void cuWrap::memDf2Dd(double *dest, const float *src, int size){
	thrust :: device_ptr <float > s ( const_cast<float *>(src) ); 
	thrust :: device_ptr <double > d ( dest); 
	thrust::transform(s, s + size , d, gpu_funcs::flt2dbl());
};
void cuWrap::memDd2Hf(float  *dest, const double  *src,int size){
	float * tmp;
	cuWrap::malloc((void **)& tmp, sizeof(float) * size);
	cuWrap::memDd2Df(tmp, src, size);
	cuWrap::memD2H(dest, tmp, sizeof(float) * size);
	cuWrap::free(tmp);

};
void cuWrap::memHd2Df(float *dest, const double *src, int size){
	double * tmp;
	cuWrap::malloc((void **)& tmp, sizeof(double) * size);
	cuWrap::memH2D(tmp, src, sizeof(double) * size);
	cuWrap::memDd2Df(dest, tmp, size);	
	cuWrap::free(tmp);

};
void cuWrap::memDd2Df(float *dest, const double *src, int size){
	thrust :: device_ptr <double > s ( const_cast<double *>(src) ); 
	thrust :: device_ptr <float > d ( dest); 
	thrust::transform(s, s + size , d, gpu_funcs::dbl2flt());
};
void cuWrap::scale(double *p, double scl, int len){
	CUBLAS_CHECK(cublasDscal(cublasHandle, len, &scl, p, 1));
};
void cuWrap::scale(float *p, float scl, int len){
	CUBLAS_CHECK(cublasSscal(cublasHandle, len, &scl, p, 1));
};
void cuWrap::plusFloatMat(double *dest, double sclD, const float *src, float sclS, int size){
	thrust :: device_ptr <float > s ( const_cast<float *>(src) ); 
	thrust :: device_ptr <double > d ( dest); 
	thrust::transform(d, d + size , s, d, gpu_funcs::plusFloatMat(sclD, sclS));
};
void cuWrap::copy(int len, float *x, int incx, float * y, int incy){
	CUBLAS_CHECK(cublasScopy(cublasHandle, len, x, incx, y, incy));
};
void cuWrap::copy(int len, double *x, int incx, double * y, int incy){
	CUBLAS_CHECK(cublasDcopy(cublasHandle, len, x, incx, y, incy));
};
float cuWrap::max_element(float *p, int len){
	thrust :: device_ptr <float > v( p ); 
	return thrust::max_element(v, v + len)[0];
};
float cuWrap::min_element(float *p, int len){
	thrust :: device_ptr <float > v( p ); 
	return thrust::min_element(v, v + len)[0];
};
double cuWrap::max_element(double *p, int len){
	thrust :: device_ptr <double > v( p ); 
	return thrust::max_element(v, v + len)[0];
};
double cuWrap::min_element(double *p, int len){
	thrust :: device_ptr <double > v( p ); 
	return thrust::min_element(v, v + len)[0];
};
void cuWrap::abs(float *p,  int len){
	thrust :: device_ptr <float> f(p); 
	thrust :: transform (f, f + len ,f , gpu_funcs::absf<float>());
};
void cuWrap::abs(double *p,  int len){
	thrust :: device_ptr <double> f(p); 
	thrust :: transform (f, f + len ,f , gpu_funcs::absf<double>());
};
void cuWrap::sqrt(float *p,  int len){
	thrust :: device_ptr <float> f(p); 
	thrust :: transform (f, f + len ,f , gpu_funcs::sqrt<float>());
};
void cuWrap::sqrt(double *p,  int len){
	thrust :: device_ptr <double> f(p); 
	thrust :: transform (f, f + len ,f , gpu_funcs::sqrt<double>());
};
void cuWrap::gemm(char transA, char transB, int rowA, int colB, int joint, const float *A, const float *B, float *C, float scale){
	CUBLAS_CHECK(cublasSgemm(cublasHandle, ( cublasOperation_t)transA, ( cublasOperation_t)transB, rowA, colB, joint, 
		&scale, A, (transA)?joint:rowA, B,  (transB)?colB:joint, &cuda_params::zerof, C, rowA));
};
void cuWrap::gemm(char transA, char transB, int rowA, int colB, int joint, const double *A, const double *B, double *C, double scale){
	CUBLAS_CHECK(cublasDgemm(cublasHandle, ( cublasOperation_t)transA, ( cublasOperation_t)transB, rowA, colB, joint, 
		&scale, A, (transA)?joint:rowA, B,  (transB)?colB:joint, &cuda_params::zerod, C, rowA));
};
void cuWrap::plus(char transA, char transB, int row, int col,  const float *A, const float *B, float *C, float A_scale, float B_scale){
	CUBLAS_CHECK(cublasSgeam(cublasHandle, ( cublasOperation_t)transA, ( cublasOperation_t)transB,  row, col, 
		&A_scale, A, transA?col:row,  &B_scale, B,  transB?col:row, C, row));
};
void cuWrap::plus(char transA, char transB, int row, int col,  const double *A, const double *B, double *C, double A_scale, double B_scale){
	CUBLAS_CHECK(cublasDgeam(cublasHandle, ( cublasOperation_t)transA, ( cublasOperation_t)transB,  row, col, 
		&A_scale, A, transA?col:row,  &B_scale, B,  transB?col:row, C, row));
};
void cuWrap::transpose(int src_row, int src_col, float *mat, int size){
	float * tmp;
	cuWrap::malloc((void **)&tmp, sizeof(float) * size);
	memD2D(tmp, mat, sizeof(float) * size);
	CUBLAS_CHECK(cublasSgeam(cublasHandle,  CUBLAS_OP_T,  CUBLAS_OP_N, src_col, src_row, 
		&cuda_params::onef, tmp, src_row,  &cuda_params::zerof, mat,  src_col, mat, src_col));
	cuWrap::free(tmp);
};
void cuWrap::transpose(int src_row, int src_col,  double *mat, int size){
	double * tmp;
	cuWrap::malloc((void **)&tmp, sizeof(double) * size);
	memD2D(tmp, mat, sizeof(double) * size);
	CUDA_CHECK;
	CUBLAS_CHECK(cublasDgeam(cublasHandle,  CUBLAS_OP_T,  CUBLAS_OP_N, src_col, src_row, 
		&cuda_params::oned, tmp, src_row,  &cuda_params::zerod, mat,  src_col, mat, src_col));
	cuWrap::free(tmp);
};
void cuWrap::product(float *A, float *B, float *ret, int size){
	thrust :: device_ptr <float > a ( A ); 
	thrust :: device_ptr <float > b ( B ); 
	thrust :: device_ptr <float > r (ret ); 
	thrust::transform(a, a + size , b, r, gpu_funcs::product<float>());
};
void cuWrap::product(double *A,  double *B, double * ret, int size){
	thrust :: device_ptr <double > a ( A ); 
	thrust :: device_ptr <double > b ( B ); 
	thrust :: device_ptr <double > r (ret ); 
	thrust::transform(a, a + size , b, r, gpu_funcs::product<double>());
};
void cuWrap::quotient(float *A, float *B, float *ret, int size){
	thrust :: device_ptr <float > a ( A ); 
	thrust :: device_ptr <float > b ( B ); 
	thrust :: device_ptr <float > r (ret ); 
	thrust::transform(a, a + size , b, r, gpu_funcs::quotient<float>());
};
void cuWrap::quotient(double *A,  double *B, double * ret, int size){
	thrust :: device_ptr <double > a ( A ); 
	thrust :: device_ptr <double > b ( B ); 
	thrust :: device_ptr <double > r (ret ); 
	thrust::transform(a, a + size , b, r, gpu_funcs::quotient<double>());
};
void cuWrap::cwiseInverse(float *dest, float *src, int size){
	thrust :: device_ptr <float > d ( dest ); 
	thrust :: device_ptr <float > s ( src ); 
	thrust::transform(s, s + size , d, gpu_funcs::cwiseInverse<float>());
};
void cuWrap::cwiseInverse(double *dest, double *src, int size){
	thrust :: device_ptr <double > d ( dest ); 
	thrust :: device_ptr <double > s ( src ); 
	thrust::transform(s, s + size , d, gpu_funcs::cwiseInverse<double>());

};
float cuWrap::dot(float *A, float *B, int size){
	float ret;
	CUBLAS_CHECK(cublasSdot (cublasHandle, size, A, 1, B, 1, &ret));
	return ret;
};
double cuWrap::dot(double *A, double *B, int size){
	double ret;
	CUBLAS_CHECK(cublasDdot (cublasHandle, size, A, 1, B, 1, &ret));
	return ret;
};
void cuWrap::sigm(float *arr,int dimen){
	thrust :: device_ptr <float> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_funcs::sigm<float>());	
}
void cuWrap::sigm(double *arr,int dimen){
	thrust :: device_ptr <double> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_funcs::sigm<double>());	
}
void cuWrap::tanh(float *arr,int dimen){
	thrust :: device_ptr <float> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_funcs::tanh<float>());
}
void cuWrap::tanh(double *arr,int dimen){
	thrust :: device_ptr <double> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_funcs::tanh<double>());
}
void cuWrap::sigmDeriv(float *diff, float * val, int dimen){
	thrust :: device_ptr <float> d(diff); 
	thrust :: device_ptr <float> v(val); 
	thrust :: transform (d, d+dimen , v, d, gpu_funcs::sigmDeriv<float>());	
}
void cuWrap::sigmDeriv(double *diff, double * val, int dimen){
	thrust :: device_ptr <double> d(diff); 
	thrust :: device_ptr <double> v(val); 
	thrust :: transform (d, d+dimen , v, d, gpu_funcs::sigmDeriv<double>());	
}
void cuWrap::tanhDeriv(float *diff, float * val, int dimen){
	thrust :: device_ptr <float> d(diff); 
	thrust :: device_ptr <float> v(val); 
	thrust :: transform (d, d+dimen , v, d, gpu_funcs::tanhDeriv<float>());	
}
void cuWrap::tanhDeriv(double *diff, double * val, int dimen){
	thrust :: device_ptr <double> d(diff); 
	thrust :: device_ptr <double> v(val); 
	thrust :: transform (d, d+dimen , v, d, gpu_funcs::tanhDeriv<double>());	
}
void cuWrap::square(float *arr,int dimen){
	thrust :: device_ptr <float> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_funcs::square<float>());
}
void cuWrap::square(double *arr,int dimen){
	thrust :: device_ptr <double> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_funcs::square<double>());
}
double cuWrap::sum(double *arr,int dimen){
	thrust :: device_ptr <double> f(arr); 
	return thrust :: reduce (f, f+dimen );
}
float cuWrap::sum(float *arr,int dimen){
	thrust :: device_ptr <float> f(arr); 
	return thrust :: reduce (f, f+dimen );
}
double cuWrap::norm(const double *A, int len){
	double ret;
	CUBLAS_CHECK(cublasDnrm2(cublasHandle, len, A, 1, &ret));
	return ret;
};
float cuWrap::norm(const float *A, int len){
	float ret;
	CUBLAS_CHECK(cublasSnrm2(cublasHandle, len, A, 1, &ret));
	return ret;
};
void cuWrap::rowSum(double *ret, double * src, double scale_src, char trans, int rows, int cols){
	cuda_params::change_one_len_d(rows);
	CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_N,  ( cublasOperation_t)trans, 1, cols, rows, 
		&scale_src, cuda_params::one_array_d, 1,  src,  (trans)?cols:rows, &cuda_params::zerod, ret, 1));
};
void cuWrap::rowSum(float *ret, float * src, float scale_src, char trans, int rows, int cols){
	cuda_params::change_one_len_f(rows);
	CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N,  ( cublasOperation_t)trans, 1, cols, rows, 
		&scale_src, cuda_params::one_array_f, 1, src,  (trans)?cols:rows, &cuda_params::zerof, ret, 1));
};
void cuWrap::colSum(double *ret, double * src, double scale_src, char trans, int rows, int cols){
	cuda_params::change_one_len_d(cols);
	CUBLAS_CHECK(cublasDgemm(cublasHandle,  ( cublasOperation_t)trans, CUBLAS_OP_N, rows, 1, cols,  
		&scale_src, src,  (trans)?cols:rows, cuda_params::one_array_d, cols, &cuda_params::zerod, ret, rows));
};
void cuWrap::colSum(float *ret, float * src, float scale_src, char trans, int rows, int cols){
	cuda_params::change_one_len_f(cols);
	CUBLAS_CHECK(cublasSgemm(cublasHandle,  ( cublasOperation_t)trans, CUBLAS_OP_N, rows, 1, cols,  
		&scale_src, src,  (trans)?cols:rows, cuda_params::one_array_f, cols, &cuda_params::zerof, ret, rows));
};
void cuWrap::random(float *dest, int len){
	curandGenerator_t gen;
	CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + rand()));
	CURAND_CHECK(curandGenerateUniform(gen, dest, len));
	CURAND_CHECK(curandDestroyGenerator(gen));  
};
void cuWrap::random(double *dest, int len){
	curandGenerator_t gen;
	CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + rand()));
	CURAND_CHECK(curandGenerateUniformDouble(gen, dest, len));
	CURAND_CHECK(curandDestroyGenerator(gen));  
	
};
void cuWrap::replicate(float * dest, const float *src, int rows, int cols, int vNum, int hNum){
	int size = rows * cols;
	int vRows =rows * vNum;
	int vSize = size * vNum;	
	int blocks = (size + cuda_threads_num - 1) / cuda_threads_num;
	g_replicate<<<blocks, cuda_threads_num>>>(dest, src, rows, size, vSize, vRows, vNum, hNum);
	CUDA_CHECK;
};

void cuWrap::replicate(double * dest, const double *src, int rows, int cols, int vNum, int hNum){
	int size = rows * cols;
	int vRows =rows * vNum;
	int vSize = size * vNum;	
	int blocks = (size + cuda_threads_num - 1) / cuda_threads_num;
	g_replicateDouble<<<blocks, cuda_threads_num>>>(dest, src, rows, size, vSize, vRows, vNum, hNum);
	CUDA_CHECK;
};
void cuWrap::colsMapping(float * dest, const float *src, const int *list, int listLen, int rows, int cols){
	int * gList;
	cuWrap::malloc((void **)&gList, sizeof(int) * listLen );
	memH2D(gList, list, sizeof(int) * listLen);
	int blocks = (rows + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(listLen, blocks);
	g_colsMapping<<<B, cuda_threads_num>>>(dest, src, gList, rows, cols);
	CUDA_CHECK;
	cuWrap::free(gList);
};
void cuWrap::colsMapping(double * dest,const  double *src, const int *list, int listLen, int rows, int cols){
	int * gList;
	cuWrap::malloc((void **)&gList, sizeof(int) * listLen );
	memH2D(gList, list, sizeof(int) * listLen);
	int blocks = (rows + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(listLen, blocks);
	g_colsMappingDouble<<<B, cuda_threads_num>>>(dest, src, gList, rows, cols);
	CUDA_CHECK;
	cuWrap::free(gList);
};
void cuWrap::rowsMapping(float * dest, const float *src, const int *list, int listLen, int rows, int cols){
	int * gList;
	cuWrap::malloc((void **)&gList, sizeof(int) * listLen );
	memH2D(gList, list, sizeof(int) * listLen);
	int blocks = (cols + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(listLen, blocks);
	g_rowsMapping<<<B, cuda_threads_num>>>(dest, src, gList, rows, cols, listLen);
	CUDA_CHECK;
	cuWrap::free(gList);
};
void cuWrap::rowsMapping(double * dest,const  double *src, const int *list, int listLen, int rows, int cols){
	int * gList;
	cuWrap::malloc((void **)&gList, sizeof(int) * listLen );
	memH2D(gList, list, sizeof(int) * listLen);
	int blocks = (cols + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(listLen, blocks);
	g_rowsMappingDouble<<<B, cuda_threads_num>>>(dest, src, gList, rows, cols, listLen);
	CUDA_CHECK;
	cuWrap::free(gList);
};
void cuWrap::matPlusRowVec(float *mat, const float *vec, float matScl, float vecScl, int rows, int cols){
	if(matScl != 1){
		CUBLAS_CHECK(cublasSscal(cublasHandle, rows * cols, &matScl, mat, 1));
	}
	int blocks = (rows + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(cols, blocks);
	g_MatPlusRowVec<<<B, cuda_threads_num>>>(mat, vec, vecScl,rows);
	CUDA_CHECK;
};
void cuWrap::matPlusRowVec(double *mat, const double *vec, double matScl, double vecScl, int rows, int cols){
	if(matScl != 1){
		CUBLAS_CHECK(cublasDscal(cublasHandle, rows * cols, &matScl, mat, 1));
	}
	int blocks = (rows + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(cols, blocks);
	g_MatPlusRowVecDouble<<<B, cuda_threads_num>>>(mat, vec, vecScl,rows);
	CUDA_CHECK;
}
void cuWrap::matPlusColVec(float *mat, const float *vec, float matScl, float vecScl, int rows, int cols){
	if(matScl != 1){
		CUBLAS_CHECK(cublasSscal(cublasHandle, rows * cols, &matScl, mat, 1));
	}
	int blocks = (cols + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(rows, blocks);
	g_MatPlusColVec<<<B, cuda_threads_num>>>(mat, vec, vecScl, rows, cols);
	CUDA_CHECK;
};
void cuWrap::matPlusColVec(double *mat, const double *vec, double matScl, double vecScl, int rows, int cols){
	if(matScl != 1){
		CUBLAS_CHECK(cublasDscal(cublasHandle, rows * cols, &matScl, mat, 1));
	}
	int blocks = (cols + cuda_threads_num - 1) / cuda_threads_num;
	dim3 B(rows, blocks);
	g_MatPlusColVecDouble<<<B, cuda_threads_num>>>(mat, vec, vecScl, rows, cols);
	CUDA_CHECK;
};

void cuWrap::Gre(float *dest, const  float * x, const float *y, int len){
	thrust :: device_ptr <float > a ( const_cast<float *>(x) ); 
	thrust :: device_ptr <float > b ( const_cast<float *>(y) ); 
	thrust :: device_ptr <float > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::Gre<float>());

};
void cuWrap::Gre(double *dest,const  double * x, const double *y, int len){
	thrust :: device_ptr <double > a ( const_cast<double *>(x) ); 
	thrust :: device_ptr <double > b ( const_cast<double *>(y) ); 
	thrust :: device_ptr <double > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::Gre<double>());
};
void cuWrap::GreEqu(float *dest,const  float * x, const float *y, int len){
	thrust :: device_ptr <float > a ( const_cast<float *>(x) ); 
	thrust :: device_ptr <float > b ( const_cast<float *>(y) ); 
	thrust :: device_ptr <float > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::GreEqu<float>());

};
void cuWrap::GreEqu(double *dest,const  double * x, const double *y, int len){
	thrust :: device_ptr <double > a ( const_cast<double *>(x) ); 
	thrust :: device_ptr <double > b ( const_cast<double *>(y) ); 
	thrust :: device_ptr <double > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::GreEqu<double>());
};
void cuWrap::Les(float *dest,const  float * x, const float *y, int len){
	thrust :: device_ptr <float > a ( const_cast<float *>(x) ); 
	thrust :: device_ptr <float > b ( const_cast<float *>(y) ); 
	thrust :: device_ptr <float > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::Les<float>());

};
void cuWrap::Les(double *dest,const  double * x, const double *y, int len){
	thrust :: device_ptr <double > a ( const_cast<double *>(x) ); 
	thrust :: device_ptr <double > b ( const_cast<double *>(y) ); 
	thrust :: device_ptr <double > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::Les<double>());
};
void cuWrap::LesEqu(float *dest,const  float * x, const float *y, int len){
	thrust :: device_ptr <float > a ( const_cast<float *>(x) ); 
	thrust :: device_ptr <float > b ( const_cast<float *>(y) ); 
	thrust :: device_ptr <float > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::LesEqu<float>());

};
void cuWrap::LesEqu(double *dest,const  double * x, const double *y, int len){
	thrust :: device_ptr <double > a ( const_cast<double *>(x) ); 
	thrust :: device_ptr <double > b ( const_cast<double *>(y) ); 
	thrust :: device_ptr <double > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::LesEqu<double>());
};
void cuWrap::Equ(float *dest,const  float * x, const float *y, int len){
	thrust :: device_ptr <float > a ( const_cast<float *>(x) ); 
	thrust :: device_ptr <float > b ( const_cast<float *>(y) ); 
	thrust :: device_ptr <float > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::Equ<float>());

};
void cuWrap::Equ(double *dest,const  double * x, const double *y, int len){
	thrust :: device_ptr <double > a ( const_cast<double *>(x) ); 
	thrust :: device_ptr <double > b ( const_cast<double *>(y) ); 
	thrust :: device_ptr <double > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::Equ<double>());
};
void cuWrap::NotEqu(float *dest,const  float * x, const float *y, int len){
	thrust :: device_ptr <float > a ( const_cast<float *>(x) ); 
	thrust :: device_ptr <float > b ( const_cast<float *>(y) ); 
	thrust :: device_ptr <float > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::NotEqu<float>());

};
void cuWrap::NotEqu(double *dest, const  double * x, const double *y, int len){
	thrust :: device_ptr <double > a ( const_cast<double *>(x) ); 
	thrust :: device_ptr <double > b ( const_cast<double *>(y) ); 
	thrust :: device_ptr <double > r (dest ); 
	thrust::transform(a, a + len , b, r, gpu_funcs::NotEqu<double>());
};
void cuWrap::Identity(float * dest, int rows, int cols){
	cuWrap::fill(dest, 0, rows * cols);
	int num = min(rows, cols);
	cuda_params::change_one_len_f(num);
	CUBLAS_CHECK(cublasScopy(cublasHandle, num, cuda_params::one_array_f, 1, dest, rows + 1));
};
void cuWrap::Identity(double * dest, int rows, int cols){
	cuWrap::fill(dest, 0, rows * cols);
	int num = min(rows, cols);
	cuda_params::change_one_len_d(num);
	CUBLAS_CHECK(cublasDcopy(cublasHandle, num, cuda_params::one_array_d, 1, dest, rows + 1));
};
void cuWrap::diagonal(float * dest, const float * src, int size){
	cuWrap::fill(dest, 0, size * size);
	CUBLAS_CHECK(cublasScopy(cublasHandle, size, src, 1, dest, size + 1));
};
void cuWrap::diagonal(double * dest, const double * src, int size){
	cuWrap::fill(dest, 0, size * size);
	CUBLAS_CHECK(cublasDcopy(cublasHandle, size, src, 1, dest, size + 1));
};
void cuWrap::inverse(float * invMat, const float * mat, int size){
	int * info ;
	int * pivo;
	cuWrap::malloc((void **) & info, sizeof(int) );
	cuWrap::malloc((void **) & pivo, sizeof(int) * size);
	float * m;
	cuWrap::malloc((void **) & m, sizeof(float) * size * size);
	cuWrap::memD2D(m, mat, sizeof(float) * size * size);
	float ** mpg;
	cuWrap::malloc((void **) & mpg, sizeof(float *));
	cuWrap::memH2D(mpg, &m, sizeof(float *));
	CUDA_CHECK;
	CUBLAS_CHECK(cublasSgetrfBatched(cublasHandle, size, mpg, size, pivo, info, 1));	
	CUDA_CHECK;
	float ** impg;
	cuWrap::malloc((void **) & impg, sizeof(float *));
	cuWrap::memH2D(impg, &invMat, sizeof(float *));
	CUDA_CHECK;
	const float ** cmpg;
	cuWrap::malloc((void **) & cmpg, sizeof(float *));
	cuWrap::memD2D(cmpg, mpg, sizeof(float *));
	CUDA_CHECK;
	cublasSgetriBatched(cublasHandle, size, cmpg, size,  pivo, impg, size, info, 1);	
	CUDA_CHECK;

	// CUBLAS_CHECK(cublasSmatinvBatched(cublasHandle, size, cmpg, 1, impg, 1, info, 1));
	cuWrap::free(info);
	cuWrap::free(pivo);
	cuWrap::free(m);
	cuWrap::free(mpg);
	cuWrap::free(cmpg);
	cuWrap::free(impg);

}; 
void cuWrap::test(){

}