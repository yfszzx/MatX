template <typename TYPE, bool CUDA>
class  matCore{
	friend matCore<TYPE, !CUDA>;	
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
	inline void load(const matMem<TYPE, CUDA> * m);
	inline void load(const matMem<TYPE, !CUDA> * m);
	inline void load(const TYPE * src, bool cuda);	
	
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
	 matCore();	 
	~matCore();
	void init(int rows, int cols);
	inline void copy(const MatriX<TYPE, CUDA> &m);
	inline void copy(const MatriX<TYPE, !CUDA> &m);
	inline void copy(const MatriX<TYPE, CUDA> &m, TYPE * prt);
	inline void tmpcopy(const MatriX<TYPE, CUDA> &m);
	inline void memRealise();
	inline void copyRealise(bool sclRealise, bool trnRealise = false);
	inline bool fix() const;	
	inline void loadMat(const TYPE * src, bool cuda = false);
	void setFix(TYPE * prt);
	void fixFree();
	
public:	
	int cols() const;
	int rows() const;
	int size() const;	
	string str() const;
	//TYPE * data();
	
};

