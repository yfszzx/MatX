template <typename TYPE, bool CUDA>
class  matCore{
	friend matCore<double, CUDA>;
	friend matCore<float, CUDA>;
	friend matCore<double, !CUDA>;
	friend matCore<float, !CUDA>;
private:
	matMem<TYPE, CUDA> * mem;
	int rowsNum;
	int colsNum;
	int sizeNum;	
	bool transFlag;		
	void free();	
	inline bool unique() const;	
	
	inline void transposeRealise();
	inline void scaleRealise();
	inline void quoteLoad(const matMem<TYPE, CUDA> * m);
	inline void load(const float * src, bool cuda);	
	inline void load(const double * src, bool cuda);	

protected:	
	TYPE scale;
	inline eigenMat eMat() const;
	inline eigenMat & eMat();
	inline TYPE * dataPrt() const;
	inline TYPE * dataPrt();
	inline int realCols() const;
	inline int realRows() const;
	inline bool trans() const;
	inline void setTrans();
	inline void setTrans(bool flag);
	inline bool isVect() const;
	inline void transMem();
	 matCore();	 
	~matCore();
	void init(int rows, int cols);
	inline void copy(const MatriX<float, CUDA> &m);
	inline void copy(const MatriX<float, !CUDA> &m);
	inline void copy(const MatriX<double, CUDA> &m);
	inline void copy(const MatriX<double, !CUDA> &m);
	inline void tmpcopy(const MatriX<TYPE, CUDA> &m);
	inline void memRealise();
	inline void copyRealise(bool sclRealise, bool trnRealise = false);
	inline void loadMat(const TYPE * src, bool cuda = false);
	string str() const;
	
public:	
	int cols() const;
	int rows() const;
	int size() const;	
};

