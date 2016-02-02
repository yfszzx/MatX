template <typename TYPE, bool CUDA>
class MachineBase{
private:
	bool overedFile();
	void trainInitialize();
	void trainMain();
	MatXG * WSs;
	string machPath;	
	static void * threadTrain( void * _this);
	static void * threadMakeBatch( void * _this);
	
	int randSeeder;
	bool batchFinished;
	void loadBatch();
protected:	
	dataSetBase<TYPE, CUDA> & dt;
	int inputNum;
	int outputNum;
	virtual  int getBatchSize() = 0;	
	void initConfig();
	MatXG Ws;
	MatXG bestWs;
	int trainCount;
	float batchInitLoss;
	int batchSize;
	void initSet(int configIdx, string name, float val);
	virtual void setConfigValue(int idx, float val)= 0;
	virtual void initConfigValue(){};
	virtual void initWs(bool trainMod = true){};
	TYPE getValidLoss();
	void save(bool finished = false);
	void load(bool trainMod = true);
	virtual void saveParameters(ofstream & fl){};
	virtual void loadParameters(ifstream & fl){};
	virtual void recordFileHead(){};
	string binFileName(int idx = -1);
	void kbPause();
	virtual void train() = 0;
	virtual bool trainOperate() = 0;
	virtual void predict( MatX * _Y,  MatX * _X, int len = 1) = 0;
	ofstream rcdFile;	
	timer rcdTimer;
	float configRecorder[100];
	string configName[100];
public:
	
	MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path);
	~MachineBase();
	void trainRun();
	void wholeValidsResult();	
	void initPredict();
	void Predict(MatX * _Y, MatX * _X, int seriesLen = 1);
	void operator ()(string s, float val);	
	void showConfigSetting();
};