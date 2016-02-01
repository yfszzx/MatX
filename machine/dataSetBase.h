template <typename TYPE, bool CUDA>
class dataSetBase{
private:
	MatriX<TYPE, false> * Xhost;
	MatriX<TYPE, false> * Thost;	
	int *trainList;		
	void makeTrainAndValidList(int * &valid, int * &train);
	void initDataSpace();
	void makeValid();
protected:
	float maxCorr;
	void showValidCorrel();
	void outputValidCsv(string path, bool Continue = false);	
	bool * validBoolList;
public:
	MatriX<TYPE, false> * X0;
	MatriX<TYPE, false> * T0;
	MatriX<TYPE, CUDA> * X;
	MatriX<TYPE, CUDA> * Y;
	MatriX<TYPE, CUDA> * T;
	MatriX<TYPE, CUDA> * Xv;
	MatriX<TYPE, CUDA> * Tv;
	MatriX<TYPE, CUDA> * Yv;	
	int validFoldIdx;
	bool randBatch;
	int inputNum;
	int outputNum;
	int crossFolds;
	int dataNum;	
	int trainNum;
	int validNum;
	int seriesLen;
	int preLen;
	int  batchSize;
	activeFunctionType actFunc;
	double batchInitLoss;
	double validInitLoss;
	void loadBatch();
	void makeBatch(int size = 0);		
	virtual void makeValidList(int validIdx);
	virtual void showResult(){};
	virtual void showValidsResult(MatGroup<TYPE, CUDA> &T, MatGroup<TYPE, CUDA> &Y){};
	virtual void pauseAction(){};
	void load(int dtNum);
	void makeValid(int validIdx);
	void show();
	void init(int iptNum, int optNum,  activeFunctionType  _actFunc, bool rndBatch, int crossFolds, int seriNum = 1);
	dataSetBase();
	~dataSetBase();
	virtual void loadData() = 0;

};