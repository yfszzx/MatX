template <typename TYPE, bool CUDA>
class MachineBase{
private:
	bool overedFile();
	void trainInit();
	MatGroup<TYPE ,CUDA> * WSs;
protected:
	MatGroup<TYPE ,CUDA> Ws;
	MatGroup<TYPE ,CUDA> bestWs;
	virtual void initWs(bool trainMod = true) = 0;
	string path;
	dataSetBase<TYPE, CUDA> & dt;
	int inputNum;
	int outputNum;
	TYPE getValidLoss();
	void save(bool finished = false);
	void load(bool trainMod = true);
	virtual void saveParameters(ofstream & fl){};
	virtual void loadParameters(ifstream & fl){};
	virtual void recordFileHead(){};
	string binFileName(int idx = -1);
	ofstream rcdFile;
	timer rcdTimer;
	virtual void train() = 0;
	virtual void predict( MatriX<TYPE, CUDA> * _Y,  MatriX<TYPE, CUDA>* _X, int len = 1) = 0;	
public:
	void wholeValidsResult(string _path);	
	MachineBase(dataSetBase<TYPE, CUDA> & dtSet);
	~MachineBase();
	void trainRun(string _path);
	void initPredict();
	void Predict(MatriX<TYPE, CUDA> * _Y, MatriX<TYPE, CUDA>* _X, int seriesLen = 1);
	virtual void trainSet(int con, float val) = 0;	
};