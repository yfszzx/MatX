template <typename MACH, typename DATASET>
class MachTrainer{
private:
	bool testMod;
	vector<int> testDataList;	
	vector<float> * resultRcorder;
	vector<int> foldsList;
	vector<int> roundIdxList;
	ofstream * rcdFile;
	
	int threadsNum;
	int foldsNum;
	string machPath;
	void validate(int num);	
	void test(int num);
	void record(bool testSet = true);
	static void * threadsAction(void * prt);
	MACH * construct(int foldIdx);
	int trainInitialize();
	void predictInit();
	void readMachList(string path);
	void readMachList(vector<int> & foldIdx, vector<int> & roundIdx, string path);
protected:
	DATASET & dt;
public:		
	vector<MACH *> machList;
	MachTrainer(DATASET & dataSet, string path, int crossFolds, int threads = 1 );
	MachTrainer(DATASET & dataSet, string path);
	void operator ()(string name, float val);
	void train();
	void verify(int samplesNum, bool testSet = true);
	void setTestMod(bool mod);
	float get(string name);
	void predict();
	int getFoldsNum() const;
};

