__global__ void g_replicate(float *dest, const float *src, int rows, int size, int verticalSize, int verticalNumRows, int verticalNum, int horizontalNum){
	int coeffPos = blockIdx.x * blockDim.x + threadIdx.x;
	if(coeffPos < size){
		int rowIdx = coeffPos % rows;
		int colIdx = coeffPos / rows;
		int initPos = colIdx * verticalNumRows + rowIdx;
		int startPos;
		for(int h = 0; h< horizontalNum; h++){
			startPos = verticalSize * h + initPos;
			for(int v = 0; v< verticalNum; v++){
				dest[startPos + rows * v] = src[coeffPos];
			}
		}	
	}
}
__global__ void g_replicateDouble(double *dest, const double *src, int rows, int size, int verticalSize, int verticalNumRows, int verticalNum, int horizontalNum){
	int coeffPos = blockIdx.x * blockDim.x + threadIdx.x;
	if(coeffPos < size){
		int rowIdx = coeffPos % rows;
		int colIdx = coeffPos / rows;
		int initPos = colIdx * verticalNumRows + rowIdx;
		int startPos;
		for(int h = 0; h< horizontalNum; h++){
			startPos = verticalSize * h + initPos;
			for(int v = 0; v< verticalNum; v++){
				dest[startPos + rows * v] = src[coeffPos];
			}
		}	
	}
}
__global__ void g_rowsMapping(float *dest, const float *src, const int * list, int rows, int cols, int listLen){
	int idx = blockIdx.x;
	int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ int rowIdx;
	if(threadIdx.x == 0){
		rowIdx = list[idx];
	}
	__syncthreads();
	if(colIdx < cols){
		dest[colIdx * listLen + idx] =src[colIdx * rows + rowIdx];
	}
}
__global__ void g_colsMapping(float *dest, const float *src, const int * list, int rows, int cols){
	int idx = blockIdx.x;
	int rowIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ int colIdx;
	if(threadIdx.x == 0){
		colIdx = list[idx];
	}
	__syncthreads();
	if(rowIdx < rows){
		dest[idx * rows + rowIdx] =src[colIdx * rows + rowIdx];
	}
}
__global__ void g_rowsMappingDouble(double *dest, const double *src, const int * list, int rows, int cols, int listLen){
	int idx = blockIdx.x;
	int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ int rowIdx;
	if(threadIdx.x == 0){
		rowIdx = list[idx];
	}
	__syncthreads();
	if(colIdx < cols){
		dest[idx * listLen + rowIdx] =src[colIdx * rows + rowIdx];
	}
}
__global__ void g_colsMappingDouble(double *dest, const double *src, const int * list, int rows, int cols){
	int idx = blockIdx.x;
	int rowIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ int colIdx;
	if(threadIdx.x == 0){
		colIdx = list[idx];
	}
	__syncthreads();
	if(rowIdx < rows){
		dest[idx * rows + rowIdx] =src[colIdx * rows + rowIdx];
	}
}
__global__ void g_MatPlusRowVec(float *mat, const float *vec, float vecScl, int rows){
	int colIdx = blockIdx.x;
	int rowIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ float val;
	if(threadIdx.x == 0){
		val = vec[colIdx] * vecScl;
	}
	__syncthreads();
	if(rowIdx < rows){
		int pos = colIdx * rows + rowIdx;
		mat[pos] += val;
	}
}
__global__ void g_MatPlusRowVecDouble(double *mat, const double *vec, double vecScl, int rows){
	int colIdx = blockIdx.x;
	int rowIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ double val;
	if(threadIdx.x == 0){
		val = vec[colIdx] * vecScl;
	}
	__syncthreads();
	if(rowIdx < rows){
		int pos = colIdx * rows + rowIdx;
		mat[pos] += val;
	}
}
__global__ void g_MatPlusColVec(float *mat, const float *vec, float vecScl, int rows, int cols){
	int rowIdx = blockIdx.x;
	int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ float val;
	if(threadIdx.x == 0){
		val = vec[rowIdx] * vecScl;
	}
	__syncthreads();
	if(colIdx < cols){
		int pos = colIdx * rows + rowIdx;
		mat[pos] += val;
	}
}
__global__ void g_MatPlusColVecDouble(double *mat, const double *vec, double vecScl, int rows, int cols){
	int rowIdx = blockIdx.x;
	int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
	__shared__ double val;
	if(threadIdx.x == 0){
		val = vec[rowIdx] * vecScl;
	}
	__syncthreads();
	if(colIdx < cols){
		int pos = colIdx * rows + rowIdx;
		mat[pos] += val;
	}
}