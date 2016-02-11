template <typename TYPE, bool CUDA>
class dataSetBase{
	friend seriesDataBase<TYPE, CUDA>;
private:
	MatriX<TYPE, false> * Xhost;
	MatriX<TYPE, false> * Thost;	
	int *trainList;		
	void makeTrainAndValidList(int * &valid, int * &train);
	void initDataSpace();
	void makeValid();
	bool seriesMod;
protected:
	float maxCorr;
	void showValidCorrel();
	void outputValidCsv(string path, bool Continue = false);	
	bool * validBoolList;
	void set(string name, float val);
	void loadSamples(const TYPE * tX, const TYPE * tY, int _dataNum);
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
	int inputNum;
	int outputNum;
	int dataNum;	
	int trainNum;
	int validNum;	
	int  batchSize;	
	double batchInitLoss;
	double validInitLoss;
	void loadBatch();
	void makeBatch(int size = 0);		
	virtual void makeValidList(int validIdx);
	virtual void showResult(){};
	virtual void showValidsResult(MatGroup<TYPE, CUDA> &T, MatGroup<TYPE, CUDA> &Y){};
	virtual void pauseAction(){};
	void makeValid(int validIdx);
	void show();
	void init(int iptNum, int optNum);
	void loadDatas(){
		createSamples();
		if(dataNum < crossFolds){
			crossFolds = dataNum;
		}
		loatTestSamples();
	}
	void loatTestSamples(){};
	dataSetBase();
	~dataSetBase();
	virtual void createSamples() = 0;
	int preLen;
	int seriesLen;
	int crossFolds;
	bool randBatch;
	bool supervise;
	activeFunctionType actFunc;
	void operator ()(string name, float val);
};