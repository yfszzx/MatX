template <typename TYPE, bool CUDA>
class dataSetBase{
	friend seriesDataBase<TYPE, CUDA>;
private:	
	int threadsNum;
	vector<MachineBase<TYPE, CUDA> *> pretreatment;	
	MatriX<TYPE, false> * X0;
	MatriX<TYPE, false> * T0;
	MatX ** X;
	MatX ** Y;
	MatX ** T;
	MatX ** Xpre;//Ԥ������
	MatX ** Tpre;//Ԥ������	
	int dataNum;
	int * dataSize;
	double * initLoss;
	int **dataList;
	int *subDataNum;	
	int thrdIdx(int foldIdx);
	void makeData(int foldIdx, int * list, int num);
	void initDataSpace();
	void pretreat(MatX & x);
	
	int foldsNum;
	int testNum;
protected:
	
	void featureFilter(bool * featureList);
	void show();	
	void init(int iptNum, int optNum, int actf = LINEAR, bool seri = false);
	void loadDataSet(const TYPE * tX, const TYPE * tY, int _dataNum);
public:
	//�ṹ����
	bool seriesMod;
	int inputNum;
	int outputNum;
	activeFunctionType actFunc;

	//�û����ò���
	bool randBatch;
	bool pretreatFlag;
	int preLen;
	int seriesLen;
	
	void makeBatch(int foldIdx, int batchNum = 0);
	float loadBatch(int foldIdx, MatX * & X, MatX * & Y, MatX * & T);//����initLoss	
	void operator ()(string name, float val);
	void loadDataList(int foldIdx, vector<int> & list);
	void setPretreat(MachineBase<TYPE, CUDA> * pre);
	float getLoss(MatX * Y, MatX * T);
	virtual void setDataList(vector<int> & list, char mod, int foldIdx);
	dataSetBase();
	~dataSetBase();
	virtual void getDataSet() = 0;	
	
	
};