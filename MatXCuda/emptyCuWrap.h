
void cuWrap::initCuda(bool showGpuInfo) {};
void  cuWrap::malloc(void ** p, size_t size) {};
void  cuWrap::free(void * p) {};
void  cuWrap::memD2H(void  *dest, const void  *src, size_t size) {};
void  cuWrap::memH2D(void *dest, const void *src, size_t size) {};
void  cuWrap::memD2D(void *dest, const void *src, size_t size) {};
void  cuWrap::fill(float *p, float value, int len) {};
void  cuWrap::fill(double *p, double value, int len) {};
void  cuWrap::scale(float *p, float scl, int len) {};
void  cuWrap::scale(double *p, double scl, int len) {};
void  cuWrap::copy(int len, float *x, int incx, float * y, int incy) {};
void  cuWrap::copy(int len, double *x, int incx, double * y, int incy) {};
float cuWrap::max_element(float *p, int len) {
	return 0;
};
float  cuWrap::min_element(float *p, int len) {
	return 0;
};
double cuWrap::max_element(double *p, int len) {
	return 0;
};
double  cuWrap::min_element(double *p, int len) {
	return 0;
};
void  cuWrap::abs(float *p, int len) {};
void  cuWrap::abs(double *p, int len) {};
void  cuWrap::gemm(char transA, char transB, int rowA, int colB, int jiont, const float*A, const float *B, float *C, float scale) {};
void  cuWrap::gemm(char transA, char transB, int rowA, int colB, int joint, const double *A, const double *B, double *C, double scale) {};
void  cuWrap::plus(char transA, char transB, int row, int col, const float *A, const float *B, float *C, float A_scale, float B_scale) {};
void  cuWrap::plus(char transA, char transB, int row, int col, const double *A, const double *B, double *C, double A_scale, double B_scale) {};
void  cuWrap::matPlusRowVec(float *mat, const float *vec, float matScl, float vecScl, int rows, int cols) {};
void  cuWrap::matPlusRowVec(double *mat, const double *vec, double matScl, double vecScl, int rows, int cols) {};
void  cuWrap::matPlusColVec(float *mat, const float *vec, float matScl, float vecScl, int rows, int cols) {};
void  cuWrap::matPlusColVec(double *mat, const double *vec, double matScl, double vecScl, int rows, int cols) {};
void  cuWrap::transpose(int src_row, int src_col, float *mat, int size) {}
void  cuWrap::transpose(int src_row, int src_col, double *mat, int size) {};
void  cuWrap::product(float *A, float *B, float *C, int size) {};
void  cuWrap::product(double *A, double *B, double *C, int size) {};
float  cuWrap::dot(float *A, float *B, int size) {
	return 0;
};
double  cuWrap::dot(double *A, double *B, int size) {
	return 0;
};
void  cuWrap::sigm(double *A, int len) {};
void  cuWrap::sigm(float *A, int len) {};
void  cuWrap::tanh(double *A, int len) {};
void  cuWrap::tanh(float *A, int len) {};
void  cuWrap::square(double *A, int len) {};
void  cuWrap::square(float *A, int len) {};
double  cuWrap::sum(double *A, int len) {
	return 0;
};
float  cuWrap::sum(float *A, int len) {
	return 0;
};
double  cuWrap::norm(const double *A, int len) {
	return 0;
};
float  cuWrap::norm(const float *A, int len) {
	return 0;
};
void  cuWrap::replicate(float * dest, const float *src, int rows, int cols, int vNum, int hNum) {};
void  cuWrap::replicate(double * dest, const double *src, int rows, int cols, int vNum, int hNum) {};
void  cuWrap::colsMapping(float * dest, const float *src, const int *list, int listLen, int rows, int cols) {};
void  cuWrap::colsMapping(double * dest, const  double *src, const int *list, int listLen, int rows, int cols) {};
void  cuWrap::rowsMapping(float * dest, const float *src, const int *list, int listLen, int rows, int cols) {};
void  cuWrap::rowsMapping(double * dest, const  double *src, const int *list, int listLen, int rows, int cols) {};
void  cuWrap::colSum(double *ret, double * src, double scale_src, char trans, int rows, int cols) {}
void  cuWrap::colSum(float *ret, float * src, float scale_src, char trans, int rows, int cols) {};
void  cuWrap::rowSum(double *ret, double * src, double scale_src, char trans, int rows, int cols) {};
void  cuWrap::rowSum(float *ret, float * src, float scale_src, char trans, int rows, int cols) {}
void  cuWrap::random(float *dest, int len) {};
void  cuWrap::random(double *dest, int len) {};
void  cuWrap::inverse(float * invMat, const float * mat, int size) {};
void  cuWrap::test() {};
void  cuWrap::Gre(float *dest, const  float * x, const float *y, int len) {};
void  cuWrap::Gre(double *dest, const  double * x, const double *y, int len) {};
void  cuWrap::GreEqu(float *dest, const  float * x, const float *y, int len) {};
void  cuWrap::GreEqu(double *dest, const  double * x, const double *y, int len);
void  cuWrap::Les(float *dest, const  float * x, const float *y, int len) {}
void  cuWrap::Les(double *dest, const  double * x, const double *y, int len) {};
void  cuWrap::LesEqu(float *dest, const  float * x, const float *y, int len) {};
void  cuWrap::LesEqu(double *dest, const  double * x, const double *y, int len) {};
void  cuWrap::Equ(float *dest, const  float * x, const float *y, int len) {}
void  cuWrap::Equ(double *dest, const  double * x, const double *y, int len) {}
void  cuWrap::NotEqu(float *dest, const  float * x, const float *y, int len) {};
void  cuWrap::NotEqu(double *dest, const  double * x, const double *y, int len) {};


