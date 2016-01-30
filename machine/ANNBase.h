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
	float batchInitLoss;
	float finishZScale;
	int showFrequence;
	int saveFrequence;
	void step();
	void annealControll();
	void threadsTasks();
	static void * threadStep( void * _this);
	static void * threadMakeBatch( void * _this);
	void loadBatch();
	//记录训练过程的参数
	int trnCount;
	float trainLoss;
	float validLoss;
	float minValidLoss;
	void trainInit();
	void showAndRecord();	
	bool stepRecord();	
	bool finish();
	void kbPause();

protected:
	searchTool<TYPE, CUDA> Search;	
	MatGroup<TYPE ,CUDA> grads;
	MatriX<TYPE, CUDA>  * stat;
	int nodes;
	double loss;
	double dataLoss;
	int batchSize;	
	virtual void saveParameters(ofstream & fl);
	virtual void loadParameters(ifstream & fl);
	virtual void recordFileHead();
	virtual void forward() = 0;
	virtual void backward() = 0;
public:
	enum TrainSet{
		Dbg = 0, //debug
		Alg = 1, //algorithm
		LR = 2, //learning rate
		CfmRounds = 3,//confirm rounds
		Mmt = 4,//momentem
		ZScl = 5,// Z scale
		BatchTrnR = 6, //batch train rounds
		RoundsDcsR = 7, //rounds decease rate
		BatchIncsR = 8, //batch incease rate
		SaveFrq = 9, // save frequence
		ShowFrq = 10, // show frequence
		BatchSz = 11, //batch size
		maxBatchSz = 12,
		finishZScl = 13
	};
	ANNBase(int _nodes, dataSetBase<TYPE, CUDA> & dtSet);
	~ANNBase();
	virtual void trainSet(int con, float val);	
	virtual void train();
	virtual void setRegulars(TYPE * regVal) = 0;	

};
