template <typename MACH, typename DATASET>
class MachTrainer{
public:
	bool testMod;
	vector<MACH *> machList;
	vector<float> * resultRcorder;
	ofstream * rcdFile;
	DATASET & dt;
	int threadsNum;
	int foldsNum;
	string machPath;
	static void * threadsAction(void * prt){
		float * ret = new float;
		MACH & mach = *(MACH *)prt;
		*ret = mach.train();
		return ret;
	}
	MACH * construct(int foldIdx){
		return new MACH(dt, machPath, foldIdx);
	};
	void setTestMod(bool mod){
		testMod = mod;
		for(int i = 0; i < foldsNum; i++){
			machList[i]->testMod = mod;
		}
	}
	MachTrainer(DATASET & dataSet, string path, int folds, int threads = 1 ):dt(dataSet){
		threadsNum = threads;
		foldsNum = folds;
		dt.setThreadsNum(threadsNum);
		dt("foldsNum", foldsNum);
		machPath = path;
		resultRcorder = new vector<float>[foldsNum];
		for(int i = 0; i < foldsNum; i ++){
			machList.push_back(construct(i));
		}
		rcdFile = NULL;
		testMod = false;

	}
	void operator ()(string name, float val){
		for(int i = 0; i < foldsNum; i ++){
			(*machList[i])(name, val);
		}
	}
	void train(){
		if(testMod){
			dt.setThreadsNum(1);
			resultRcorder[0].clear();
			resultRcorder[0].push_back(machList[0]->train());
		}else{
			dt.setThreadsNum(threadsNum);
			pthread_t * tid = new pthread_t[threadsNum];
			void ** ret = new void *[threadsNum];
			for(int i = 0; i < (foldsNum + threadsNum -1)/threadsNum; i++){
				for(int t = 0; t < threadsNum; t++){
					int idx = i * threadsNum + t;
					if(idx == foldsNum){
						break;
					}
					pthread_create(&tid[t], NULL, threadsAction, machList[idx]);	
				}
				for(int t = 0; t < threadsNum; t++){
					int idx = i * threadsNum + t;
					if(i * threadsNum + t == foldsNum){
						break;
					}
					pthread_join(tid[t], &ret[t]);		
					resultRcorder[idx].clear();
					resultRcorder[idx].push_back(*(float *)ret[t]);
					delete ret[t];
				}
			}
		}
	}
	void validate(int num){
		if(testMod){
			vector<float> res = machList[0]->validate(num);			
			for(int j = 0; j < res.size(); j++){
				resultRcorder[0].push_back(res[j]);				
			}
			cout<<"\n";
			for(int j = 0; j < resultRcorder[0].size(); j++){
				cout<<resultRcorder[0][j]<<"\t";				
			}
		}else{
			dt.setThreadsNum(1);
			for(int i = 0; i < foldsNum; i++){			
				vector<float> res = machList[i]->validate(num);			
				for(int j = 0; j < res.size(); j++){
					resultRcorder[i].push_back(res[j]);				
				}
				cout<<"\n";
				for(int j = 0; j < resultRcorder[i].size(); j++){
					cout<<resultRcorder[i][j]<<"\t";				
				}
			}
		}
	}
	void record(int round){
		int num = resultRcorder[0].size();
		if(rcdFile == NULL){			
			rcdFile = new ofstream [num];
			for(int i = 0; i < num; i++){
				stringstream nm;
				nm<<machPath<<"rcd_"<<i<<".csv";
				rcdFile[i].open(nm.str());
			}
		}
		if(testMod){
			for(int j = 0; j < num; j++){
				rcdFile[j]<<round<<",";
				rcdFile[j]<<resultRcorder[0][j]<<",";
				rcdFile[j]<<endl;
			}
		}else{
			for(int j = 0; j < num; j++){
				rcdFile[j]<<round<<",";
				for(int i = 0; i < foldsNum; i++){	
					rcdFile[j]<<resultRcorder[i][j]<<",";
				}
				rcdFile[j]<<endl;
			}
		}

	}
	void test(int num){

	}
};