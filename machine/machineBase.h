template <typename TYPE, bool CUDA>
class MachineBase{
private:
	bool overedFile(int foldIdx = -1);
	void trainInitialize();
	void trainMain(int idx);
	void saveConfig(ofstream & fl);
	void loadConfig(ifstream & fl);
	MatXG * foldsMach;
	string machPath;	
	static void * threadTrain( void * _this);
	static void * threadMakeBatch( void * _this);	
	int randSeeder;
	bool batchFinished;
	void loadBatch();
	MatXG bestMach;
	void load(bool trainMod, int foldIdx);
	string binFileName(int idx = -1);
	bool supervise;


protected:	
	dataSetBase<TYPE, CUDA> & dt;
	int inputNum;
	int outputNum;	
	int unsuperviseDim;
	int trainCount;
	float batchInitLoss;
	int batchSize;
	MatXG Mach;
	ofstream rcdFile;	
	timer rcdTimer;
	vector<float> configRecorder;
	vector<string> configName;

	void initConfig();
	void setBestMach();	
	void initSet(int configIdx, string name, float val);
	void kbPause();
	TYPE getValidLoss();
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
	virtual void predictCore( MatX * _Y,  MatX * _X, int len = 1) = 0;
	virtual TYPE unsupervisedExamine(MatX * _Y,  MatX * _X, int len = 1){ return 0;};//用于检验非监督学习的结果
public:
	
	MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path);
	~MachineBase();
	void save(bool finished = false);
	void trainRun();	
	void predictInit();
	void predict(MatX * _Y, MatX * _X, int seriesLen = 1);
	void operator ()(string s, float val);	
	void kbGet(string name);
	void showValidsResult();
	void showConfigSetting();
	void clear();	
	inline int getUnsupDim();
	
};