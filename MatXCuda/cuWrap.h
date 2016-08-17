namespace cuWrap{
	 void initCuda(bool showGpuInfo);
	 void malloc(void ** p, size_t size);
	 void free(void * p);
	 //D mean device, H mean host
	 void memD2H(void  *dest, const void  *src, size_t size);
	 void memH2D(void *dest, const void *src, size_t size);
	 void memD2D(void *dest, const void *src, size_t size);
	 //f mean float d mean double
	 void memDf2Hd(double  *dest, const float  *src, int size);
	 void memHf2Dd(double *dest, const float *src, int size);
	 void memDf2Dd(double *dest, const float *src, int size);
	 void memHf2Hd(double *dest, const float *src, int size);
	 void memDd2Hf(float  *dest, const double  *src,int size);
	 void memHd2Df(float *dest, const double *src, int size);
	 void memDd2Df(float *dest, const double *src, int size);
	 void memHd2Hf(float *dest, const double *src, int size);
	 void plusFloatMat(double *dest, double sclD, const float *src, float sclS, int size);
	 void fill(float *p, float value, int len);
	 void fill(double *p, double value, int len);
	 void scale(float *p, float scl, int len);
	 void scale(double *p, double scl, int len);
	 void copy(int len, float *x, int incx, float * y, int incy);
	 void copy(int len, double *x, int incx, double * y, int incy);
	 float max_element(float *p, int len);
	 float min_element(float *p, int len);
	 double max_element(double *p, int len);
	 double min_element(double *p, int len);
	 void abs(float *p,  int len);
	 void abs(double *p,  int len);
	 void sqrt(float *p,  int len);
	 void sqrt(double *p,  int len);
	 //trans: 0 不转置 1 转置 2 共轭转置
	 void gemm(char transA, char transB, int rowA, int colB, int jiont, const float*A, const float *B, float *C, float scale);
	 void gemm(char transA, char transB, int rowA, int colB, int joint, const double *A, const double *B, double *C, double scale);
	 void plus(char transA, char transB, int row, int col,  const float *A, const float *B, float *C, float A_scale, float B_scale);
	 void plus(char transA, char transB, int row, int col,  const double *A, const double *B, double *C, double A_scale, double B_scale);
	 void matPlusRowVec(float *mat, const float *vec, float matScl, float vecScl, int rows, int cols);
	 void matPlusRowVec(double *mat, const double *vec, double matScl, double vecScl, int rows, int cols);
	 void matPlusColVec(float *mat, const float *vec, float matScl, float vecScl, int rows, int cols);
	 void matPlusColVec(double *mat, const double *vec, double matScl, double vecScl, int rows, int cols);
	 void transpose(int src_row, int src_col,  float *mat,  int size);
	 void transpose(int src_row, int src_col,  double *mat,  int size);
	 void product(float *A,  float *B, float *C, int size);
	 void product(double *A, double *B, double *C, int size);
	 void quotient(float *A,  float *B, float *C, int size);
	 void quotient(double *A, double *B, double *C, int size);
	 void cwiseInverse(float *dest, float *src, int size);
	 void cwiseInverse(double *dest, double *src, int size);
	 float dot(float *A,  float *B, int size);
	 double dot( double *A, double *B, int size);
	 void sigm(double *A, int len);
	 void sigm(float *A, int len);
	 void tanh(double *A, int len);
	 void tanh(float *A, int len);
	 void sigmDeriv(double *diff, double * val, int len);
	 void sigmDeriv(float *diff, float * val,int len);
	 void tanhDeriv(double *diff, double * val, int len);
	 void tanhDeriv(float *diff, float * val, int len);
	 void square(double *A, int len);
	 void square(float *A, int len);
	 double sum(double *A, int len);
	 float sum(float *A, int len);
	 double norm(const double *A, int len);
	 float norm(const float *A, int len);
	 void replicate(float * dest, const float *src, int rows, int cols, int vNum, int hNum);
	 void replicate(double * dest, const double *src, int rows, int cols, int vNum, int hNum);
	 void colsMapping(float * dest, const float *src, const int *list, int listLen, int rows, int cols);
	 void colsMapping(double * dest,const  double *src, const int *list, int listLen, int rows, int cols);
	 void rowsMapping(float * dest, const float *src, const int *list, int listLen, int rows, int cols);
	 void rowsMapping(double * dest,const  double *src, const int *list, int listLen, int rows, int cols);
	 void colSum(double *ret, double * src, double scale_src, char trans, int rows, int cols);
	 void colSum(float *ret, float * src, float scale_src, char trans, int rows, int cols);
	 void rowSum(double *ret, double * src, double scale_src, char trans, int rows, int cols);
	 void rowSum(float *ret, float * src, float scale_src, char trans, int rows, int cols);
	 void normalRandom(double *dest, int len, double mean, double dev);
	 void normalRandom(float *dest, int len, float mean, float dev);
	 void random(float *dest, int len);
	 void random(double *dest, int len);
	 void inverse(float * invMat, const float * mat, int size); 
	 void test();
	 // Gre mean greater , Equ mean equare , Les mean less
	 void Gre(float *dest,const  float * x, const float *y, int len);
	 void Gre(double *dest,const  double * x, const double *y, int len);
	 void GreEqu(float *dest,const  float * x, const float *y, int len);
	 void GreEqu(double *dest,const  double * x, const double *y, int len);
	 void Les(float *dest,const  float * x, const float *y, int len);
	 void Les(double *dest,const  double * x, const double *y, int len);
	 void LesEqu(float *dest,const  float * x, const float *y, int len);
	 void LesEqu(double *dest,const  double * x, const double *y, int len);
	 void Equ(float *dest,const  float * x, const float *y, int len);
	 void Equ(double *dest,const  double * x, const double *y, int len);
	 void NotEqu(float *dest,const  float * x, const float *y, int len);
	 void NotEqu(double *dest,const  double * x, const double *y, int len);	
	 void Identity(float * dest, int rows, int cols);
	 void Identity(double * dest, int rows, int cols);
	 void diagonal(float * dest, const float * src, int size);
	 void diagonal(double * dest, const double * src, int size);

	 


	 



















};
