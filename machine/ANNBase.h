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
	

	void step();
	void annealControll();
	void threadsTasks();
	static void * threadStep( void * _this);
	static void * threadMakeBatch( void * _this);
	void loadBatch();
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
	virtual void initWs(bool trainMod = true);
	virtual void saveParameters(ofstream & fl);
	virtual void loadParameters(ifstream & fl);
	virtual void recordFileHead();
	virtual int getBatchSize();
	virtual bool trainOperate();
	virtual void train();
	void initConfigValue();
	void setConfigValue(int idx, float val);
	void annInitConfig();
	virtual void annInitConfigValue() = 0;
	virtual void annSetConfigValue(int idx, float val) = 0;
	virtual void annInitWs(bool trainMod) = 0;
	virtual void predict( MatX * _Y,  MatX* _X, int len = 1) = 0;
	virtual void forward() = 0;
	virtual void backward() = 0;
public:
	ANNBase(dataSetBase<TYPE, CUDA> & dtSet, string path);
	
	
};
