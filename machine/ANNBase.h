template <typename TYPE, bool CUDA>
class ANNBase:public MachineBase<TYPE, CUDA>{	
private:

	float initBatchTrainRounds;
	float initBatchSize;
	float batchTrainRounds;
	float batchSizeControlar;
	float maxBatchSize;
	float roundsDeceaseRate;
	float batchInceaseRate;
	float finishZScale;
	int showFrequence;
	int saveFrequence;
	//记录训练过程的参数
	float trainLoss;
	float validLoss;
	float minValidLoss;

	bool breakFlag(int cnt);
	void step();
	void annealControll();
	void threadsTasks();
	void showAndRecord();	
	bool stepRecord();	
	bool finish();
	int subConfigsNum;
protected:
	searchTool<TYPE, CUDA> Search;	
	MatXG grads;
	double loss;
	double dataLoss;
	string annConfigName[100];
	virtual void saveParameters(ofstream & fl);
	virtual void loadParameters(ifstream & fl);
	virtual void recordFileHead();
	virtual int getBatchSize();
	virtual void trainHead();
	virtual void trainCore();
	virtual bool trainAssist();
	void initConfigValue();
	void setConfigValue(int idx, float val);
	virtual void annInitConfigValue() = 0;
	virtual void annSetConfigValue(int idx, float val) = 0;
	virtual void initMachine() = 0;
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1) = 0;
	virtual void annTrainHead() = 0;
	virtual void forward() = 0;
	virtual void backward() = 0;
public:
	ANNBase(dataSetBase<TYPE, CUDA> & dtSet, string path);	
};
