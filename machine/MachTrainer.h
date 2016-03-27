template <typename MACH, typename DATASET>
class MachTrainer{
private:
	bool testMod;
	vector<int> testDataList;
	
	vector<float> * resultRcorder;
	ofstream * rcdFile;
	DATASET & dt;
	int threadsNum;
	int foldsNum;
	string machPath;
	void validate(int num);	
	void test(int num);
	void record();
	static void * threadsAction(void * prt);
	MACH * construct(int foldIdx);
public:		
	vector<MACH *> machList;
	MachTrainer(DATASET & dataSet, string path, int crossFolds, int threads = 1 );
	void operator ()(string name, float val);
	void train();
	void verify(int samplesNum, bool testSet = true);
	void setTestMod(bool mod);
	float get(string name);
};