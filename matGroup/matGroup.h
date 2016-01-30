template <typename TYPE, bool CUDA >
class MatGroup{
	friend MatGroup <TYPE, !CUDA>;
private:
	MatriX<TYPE, CUDA> ** mats;
	MatriX<TYPE, CUDA> fixMemMat;
	int matsNum;
	bool selfSpace;
	bool fixMat;
	
	void copy(const MatGroup<TYPE, CUDA> & m);
	void copyFix(const MatGroup<TYPE, CUDA> & m);
	void memFree();
public:
	MatGroup();
	~MatGroup();
	MatGroup(int num);
	MatGroup<TYPE, CUDA>(const MatGroup<TYPE, CUDA> & m);
	MatGroup<TYPE, CUDA>(const MatGroup<TYPE, !CUDA> & m);
	MatGroup<TYPE, CUDA>(MatriX<TYPE, CUDA> * m, int num);
	MatGroup<TYPE, CUDA>(MatriX<TYPE, !CUDA> * m, int num);
	MatGroup<TYPE, CUDA> & operator <<(MatriX<TYPE, CUDA> & mat);
	MatGroup<TYPE, CUDA> & operator += (const MatGroup<TYPE, CUDA> & m);
	MatGroup<TYPE, CUDA> & operator -= (const MatGroup<TYPE, CUDA> & m);
	MatGroup<TYPE, CUDA>  operator + (const MatGroup<TYPE, CUDA> & m) const;
	MatGroup<TYPE, CUDA>  operator - (const MatGroup<TYPE, CUDA> & m) const;
	MatGroup<TYPE, CUDA> & operator = (const MatGroup<TYPE, CUDA> & m);
	MatGroup<TYPE, CUDA> & operator /= ( TYPE scl);
	MatGroup<TYPE, CUDA> & operator *= ( TYPE scl);
	MatGroup<TYPE, CUDA> & operator = ( TYPE scl);
	MatGroup<TYPE, CUDA>  operator * ( TYPE scl) const;
	MatGroup<TYPE, CUDA>  operator / ( TYPE scl) const;
	MatriX<TYPE, CUDA> & operator [] (int i) const;
	MatGroup<TYPE, CUDA> & clear();
	TYPE squaredNorm() const;
	TYPE norm() const;
	TYPE dot(const MatGroup<TYPE, CUDA> & m) const;
	TYPE correl(const MatGroup<TYPE, CUDA> & m) const;
	TYPE sum() const;
	TYPE MSE() const;
	void show() const;
	void save(ofstream & fl) const;
	void read(ifstream & fl) ;
	void setFix();
	void fixFree();
	int num() const;
	int size() const;

};
template <typename TYPE,bool CUDA >
MatGroup<TYPE, CUDA>  operator * (const TYPE  scl, const MatGroup<TYPE, CUDA> & m);
template <typename TYPE,bool CUDA >
MatGroup<TYPE, CUDA>  operator - (const MatGroup<TYPE, CUDA> & m);