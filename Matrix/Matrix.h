template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> operator * (TYPE val, const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> operator + (TYPE val, const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> operator - (TYPE val, const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> sigm(const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> tanh(const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> square(const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> abs(const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
inline MatriX<TYPE, CUDA> sqrt(const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> operator -(const MatriX<TYPE, CUDA> &m);
template <typename TYPE, bool CUDA>
ostream& operator <<(ostream &os, const MatriX<TYPE, CUDA> &m);

template <typename TYPE, bool CUDA>
class MatriX : public matCore<TYPE, CUDA> {
	friend class MatriX<TYPE, !CUDA>;
	friend matCore<float, !CUDA>;
	friend matCore<float, CUDA>;
	friend matCore<double, !CUDA>;
	friend matCore<double, CUDA>;
	friend MatriX<double, CUDA>;
	friend MatriX<float, CUDA>;
	friend MatriX<TYPE, CUDA> operator * <TYPE, CUDA>(TYPE val, const MatriX<TYPE, CUDA> &m);
	friend MatriX<TYPE, CUDA> operator + <TYPE, CUDA>(TYPE val, const MatriX<TYPE, CUDA> &m);
	friend MatriX<TYPE, CUDA> operator - <TYPE, CUDA>(TYPE val, const MatriX<TYPE, CUDA> &m);
	friend MatriX<TYPE, CUDA> sigm <TYPE, CUDA>(const MatriX<TYPE, CUDA> &m);
	friend MatriX<TYPE, CUDA> tanh <TYPE, CUDA>(const MatriX<TYPE, CUDA> &m);
	friend  MatriX<TYPE, CUDA> square <TYPE, CUDA>(const MatriX<TYPE, CUDA> &m);
	friend  MatriX<TYPE, CUDA> abs <TYPE, CUDA>(const MatriX<TYPE, CUDA> &m);
	friend  MatriX<TYPE, CUDA> sqrt <TYPE, CUDA>(const MatriX<TYPE, CUDA> &m);
	friend ostream& operator << <TYPE, CUDA>(ostream &os, const MatriX<TYPE, CUDA> &m);
	friend MatriX<TYPE, CUDA> operator - <TYPE, CUDA>(const MatriX<TYPE, CUDA> &m);
	friend MatGroup<TYPE, CUDA>;
private:
	enum operateType { STRU = 1, CON = 2, TRAN = 3, SCL = 4, ALL = 5 ,TURN = 6};
	//STRU 开辟空间，拷贝结构;  CON 完全拷贝; TRAN 拷贝，若tranFlag == true 则实现转置; SCL 拷贝并实现scale; ALL 联合执行 TRAN 和SCL;TURN 将矩阵内存转置,外观不变
	static bool sortCmp(int a, int b);
	void removeMapping(vector<int> & map, int * list, const int n, const int remove_num) const;
	inline int elementPos(int rows, int cols) const;
	inline int realPos(int idx) const;
	inline bool matPlusVec(const MatriX<TYPE, CUDA> &vec);
	MatriX(const MatriX<TYPE, CUDA> &m, operateType type);

protected:
	//常用函数
	inline MatriX & tanh();
	inline MatriX & sigm();
	inline MatriX & square();
	inline MatriX & abs();
	inline MatriX & sqrt();
public:

	MatriX(int _rows = 0, int _cols = 1);
	MatriX(const MatriX<double, CUDA> &m);
	MatriX(const MatriX<double, !CUDA> &m);
	MatriX(const MatriX<float, CUDA> &m);
	MatriX(const MatriX<float, !CUDA> &m);


	//复制
	MatriX &  operator = (const MatriX<double, CUDA> &m);
	MatriX &  operator = (const MatriX<double, !CUDA> &m);
	MatriX &  operator = (const MatriX<float, CUDA> &m);
	MatriX &  operator = (const MatriX<float, !CUDA> &m);
	MatriX & operator = (TYPE val);
	MatriX & selfRandom();
	static  MatriX<TYPE, CUDA> Random(int _rows, int  _cols = 1);
	static  MatriX<TYPE, CUDA> Constant(int _rows, int  _cols, TYPE val);
	static  MatriX<TYPE, CUDA> Ones(int _rows, int  _cols = 1);
	static  MatriX<TYPE, CUDA> Zero(int _rows, int  _cols = 1);
	static  MatriX<TYPE, CUDA> Identity(int rows, int cols = 0);
	static  MatriX<TYPE, CUDA> eye(int rows, int cols = 0);
	static  MatriX<TYPE, CUDA> Diagonal(const  MatriX<TYPE, CUDA> & mat);
	inline MatriX<TYPE, CUDA> & assignment(int row, int col, TYPE val);
	inline MatriX<TYPE, CUDA> & assignment(int idx, TYPE val);
	



	//基本运算
	inline MatriX transpose() const;
	inline MatriX T() const;
	inline MatriX operator * (const MatriX<TYPE, CUDA> &mat) const;
	inline MatriX operator * (TYPE val) const;
	inline MatriX & operator *= (TYPE val);
	inline MatriX & operator /= (TYPE val);
	inline MatriX & operator += (TYPE val);
	inline MatriX & operator -= (TYPE val);
	inline MatriX operator / (TYPE val) const;
	inline MatriX operator + (const MatriX<TYPE, CUDA> &mat) const;
	inline MatriX operator - (const MatriX<TYPE, CUDA> &mat) const;
	inline MatriX operator + (TYPE val) const;
	inline MatriX operator - (TYPE val) const;
	inline MatriX & operator += (const MatriX<TYPE, CUDA> &mat);
	inline MatriX & operator -= (const MatriX<TYPE, CUDA> &mat);
	MatriX cwiseProduct(const MatriX<TYPE, CUDA> &mat) const;
	MatriX cwiseQuotient(const MatriX<TYPE, CUDA> &mat) const;
	MatriX cwiseInverse() const;
	MatriX rankUpdate() const; /*未完成*/
	//加入矩阵与对角矩阵乘法
	inline TYPE dot(const MatriX<TYPE, CUDA> &mat) const;	
	inline MatriX inverse() const;
	inline MatriX inv() const;
	inline MatriX & add(const MatriX<float, CUDA> &mat);
	inline MatriX & add(const MatriX<double, CUDA> &mat);

	MatriX<TYPE, CUDA> operator > (const MatriX<TYPE, CUDA> & m)  const;
	MatriX<TYPE, CUDA> operator < (const MatriX<TYPE, CUDA> & m)  const;
	MatriX<TYPE, CUDA> operator >= (const MatriX<TYPE, CUDA> & m)  const;
	MatriX<TYPE, CUDA> operator <= (const MatriX<TYPE, CUDA> & m)  const;
	MatriX<TYPE, CUDA> operator == (const MatriX<TYPE, CUDA> & m)  const;
	MatriX<TYPE, CUDA> operator != (const MatriX<TYPE, CUDA> & m)  const;


	//mapping
	TYPE operator [] (int idx)  const;
	TYPE operator () (int rowIdx, int colIdx)  const;
	inline MatriX replicate(int vertical_times, int horizontal_times)  const;
	MatriX<TYPE, CUDA> row(int _row)  const;
	MatriX<TYPE, CUDA> col(int _col)  const;
	MatriX<TYPE, CUDA> topRows(int num = 1)  const;
	MatriX<TYPE, CUDA> bottomRows(int num = 1)  const;
	MatriX<TYPE, CUDA> leftCols(int num = 1)  const;
	MatriX<TYPE, CUDA> rightCols(int num = 1)  const;
	MatriX<TYPE, CUDA> rowsMapping(int *list, int num) const;
	MatriX<TYPE, CUDA> colsMapping(int *list, int num) const;
	MatriX<TYPE, CUDA> removeTopRows(int num = 1)  const;
	MatriX<TYPE, CUDA> removeBottomRows(int num = 1)  const;
	MatriX<TYPE, CUDA> removeLeftCols(int num = 1)  const;
	MatriX<TYPE, CUDA> removeRightCols(int num = 1)  const;
	MatriX<TYPE, CUDA> removeCols(int * list, int num)  const;
	MatriX<TYPE, CUDA> removeRows(int * list, int num)  const;
	MatriX<TYPE, CUDA> removeCol(int idx)  const;
	MatriX<TYPE, CUDA> removeRow(int idx)  const;
	MatriX<TYPE, CUDA> colJoint(const MatriX<TYPE, CUDA> & mat)  const;
	MatriX<TYPE, CUDA> rowJoint(const MatriX<TYPE, CUDA> & mat)  const;


	//statistics

	inline TYPE allSum() const;
	inline TYPE allMean()  const;
	inline TYPE allMSE()  const;
	inline TYPE allMSE(TYPE & avg)  const;
	inline TYPE norm() const;
	inline TYPE squaredNorm() const;
	inline TYPE norm2() const;
	inline MatriX<TYPE, CUDA> sum() const;
	inline MatriX<TYPE, CUDA> mean()  const;
	inline MatriX<TYPE, CUDA> MSE(MatriX<TYPE, CUDA>& avg)  const;
	inline MatriX<TYPE, CUDA> MSE()  const;
	inline MatriX<TYPE, CUDA> RMS(MatriX<TYPE, CUDA>& avg)  const;
	inline MatriX<TYPE, CUDA> RMS()  const;
	inline MatriX<TYPE, CUDA> rowsSum()  const;//效率待改进
	inline MatriX<TYPE, CUDA> colsSum()  const;//效率待改进
	TYPE correl(MatriX<TYPE, CUDA>& mat) const;
	TYPE allMax() const;
	TYPE allMin() const;

	//others
	MatriX<TYPE, CUDA> &importData(TYPE * src, bool cuda = false);
	void exportData(TYPE * & dt, bool cuda = false) const;
	void save(ofstream & fl) const;
	void read(ifstream & fl);
	string str(){
		return matCore::str();
	}

	//eigenvalue
	MatriX<TYPE, CUDA> eigenValues() const;
	MatriX<TYPE, CUDA> eigenSolver(MatriX<TYPE, CUDA>& eigenVals) const;
	TYPE spectralRadius() const;
};


