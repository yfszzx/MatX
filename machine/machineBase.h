template <typename TYPE, bool CUDA>
class MachineBase{
private:
	bool overedFile(int foldIdx = -1);
	void trainInitialize();
	void trainMain();
	MatXG * foldsMach;
	string machPath;	
	static void * threadTrain( void * _this);
	static void * threadMakeBatch( void * _this);
	
	int randSeeder;
	bool batchFinished;
	void loadBatch();
	MatXG bestMach;
protected:	
	dataSetBase<TYPE, CUDA> & dt;
	int inputNum;
	int outputNum;	
	void initConfig();
	MatXG Mach;
	void setBestMach(const MatXG & best);
	int trainCount;
	float batchInitLoss;
	int batchSize;
	void initSet(int configIdx, string name, float val);
	void kbPause();
	TYPE getValidLoss();
	void save(bool finished = false);
	void load(bool trainMod = true);
	string binFileName(int idx = -1);	
	void predictInit();
	ofstream rcdFile;	
	timer rcdTimer;
	float configRecorder[100];
	string configName[100];

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
	virtual void predictCore( MatX * _Y,  MatX * _X, int len = 1) = 0;
public:
	
	MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path);
	~MachineBase();
	void trainRun();
	void wholeValidsResult();		
	void predict(MatX * _Y, MatX * _X, int seriesLen = 1);
	void operator ()(string s, float val);	
	void showConfigSetting();
};