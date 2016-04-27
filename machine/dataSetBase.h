template <typename TYPE, bool CUDA>
class dataSetBase{
	friend seriesDataBase<TYPE, CUDA>;
	
protected:
	int ** batchList;
	vector<MachineBase<TYPE, CUDA> *> pretreatment;	
	MatriX<TYPE, false> * X0;
	MatriX<TYPE, false> * T0;
	
	int * dataSize;
	double * initLoss;
	int **dataList;
	int *subDataNum;	

	void makeData(int foldIdx);
	void initDataSpace();
	void pretreat(MatX & x);

	int testNum;
	int dataNum;
	int foldsNum;	
	void featureFilter(bool * featureList);
	int thrdIdx(int foldIdx);	
	void init(int iptNum, int optNum, int actf = LINEAR);
	void loadDataSet(const TYPE * tX, const TYPE * tY, int _dataNum);
	virtual bool trainSample(int idx, int foldIdx);
	virtual bool validSample(int idx, int foldIdx);
	virtual bool testSample(int idx,  int foldIdx);
public:
	//结构参数
	bool seriesMod;
	int inputNum;
	int outputNum;
	activeFunctionType actFunc;

	//用户设置参数
	bool randBatch;
	bool pretreatFlag;
	int preLen;
	int seriesLen;
	int threadsNum;
	MatX ** X;
	MatX ** Y;
	MatX ** T;
	MatX ** Xpre;//预备数据
	MatX ** Tpre;//预备数据	
	MatX * forecastX;
	MatX * forecastY;
	MatXD * forecastSumY;
	TYPE * forecastOutput;
	int forecastLen;

	bool testFold;
	int roundIdx;


	void setDataList(vector<int> & list, int mod, int foldIdx);
	void makeBatch(int foldIdx, int batchNum = 0, bool replacement = true);
	float loadBatch(int foldIdx, MatX * & X, MatX * & Y, MatX * & T);//返回initLoss	
	void operator ()(string name, float val);
	void loadDataList(int foldIdx, vector<int> & list);
	void setPretreat(MachineBase<TYPE, CUDA> * pre);
	float getLoss(int foldIdx);
	float getCorrel(int foldIdx);
	virtual void getMoreResult(vector<float> & res, int foldIdx){};
	
	dataSetBase();
	~dataSetBase();
	virtual void getDataSet(){};	
	vector<float> getResult(int foldIdx);


	void show();	
	void setThreadsNum(int num);
	int getDataNum(int foldIdx);	
	void load(TYPE * Y, const TYPE * X, int _dataNum);
};