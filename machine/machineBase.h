template <typename TYPE, bool CUDA>
class MachineBase{
private:
	string machPath;	
	int randSeeder;
	vector<int> trainDataList;
	vector<int> validDataList;
	vector<int> dataList;
	void trainInitialize();
	void trainFinished();
	void saveConfig(ofstream & fl);
	void loadConfig(ifstream & fl);

	static void * threadTrain( void * _this);
	static void * threadMakeBatch( void * _this);		
	void loadBatch();
	void load(int roundIdx);
	void save(int roundIdx);
	string binFileName(int roundIdx);
	bool supervise;
	bool finishFlag;
	int foldIdx;
	int trainRoundIdx;
	void loadMach(int rndIdx);
protected:	
	dataSetBase<TYPE, CUDA> & dt;
	int inputNum;
	int outputNum;	
	int unsuperviseDim;
	int trainCount;
	float batchInitLoss;
	int batchSize;
	MatXG Mach;
	MatX * X;
	MatX * Y;
	MatX * T;
	ofstream rcdFile;	
	timer rcdTimer;
	vector<float> configRecorder;
	vector<string> configName;

	void initConfig();
	void initSet(int configIdx, string name, float val);	
	void unsupervise();
	virtual  int getBatchSize() = 0;	
	virtual void saveParameters(ofstream & fl){};
	virtual void loadParameters(ifstream & fl){};
	virtual void recordFileHead(){};
	virtual void initConfigValue(){};
	virtual void setConfigValue(int idx, float val){};
	virtual void initMachine() = 0;
	virtual void trainHead(){};
	virtual void trainCore() = 0;
	virtual bool trainAssist(){return 1;};
	virtual void trainTail(){};
	virtual void predictHead(){};	
	virtual TYPE unsupervisedExamine(MatX * _Y,  MatX * _X, int len = 1){ return 0;};//用于检验非监督学习的结果
	TYPE getValidLoss();
	void kbGet(string name);
	
public:

	MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path, int foldIdx);
	~MachineBase();	
	void operator ()(string s, float val);	
	void showConfigSetting();
	void train();
	void validate(int validNum);
	int getUnsupDim();
	virtual void predict( MatX * _Y,  MatX * _X, int len = 1) = 0;
	void predictInit();
};

