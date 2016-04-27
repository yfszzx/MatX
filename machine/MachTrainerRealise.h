template <typename MACH, typename DATASET>
void * MachTrainer<MACH, DATASET>::threadsAction(void * prt){
	float * ret = new float;
	MACH & mach = *(MACH *)prt;
	*ret = mach.train();
	return ret;
}
template <typename MACH, typename DATASET>
MACH * MachTrainer<MACH, DATASET>::construct(int foldIdx){
	return new MACH(dt, machPath, foldIdx);
};
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::setTestMod(bool mod){
	testMod = mod;
	for(int i = 0; i < foldsNum; i++){
		machList[i]->testMod = mod;
	}
}
template <typename MACH, typename DATASET>
float MachTrainer<MACH, DATASET>::get(string name){
	return machList[0]->get(name);
}
template <typename MACH, typename DATASET>
 MachTrainer<MACH, DATASET>::MachTrainer(DATASET & dataSet, string path, int crossFolds, int threads):dt(dataSet){
	threadsNum = threads;
	foldsNum = crossFolds;
	dt.setThreadsNum(threadsNum);
	dt("foldsNum", foldsNum);
	machPath = path;
	resultRcorder = new vector<float>[foldsNum + 1];
	for(int i = 0; i < foldsNum; i ++){
		machList.push_back(construct(i));
	}
	rcdFile = NULL;
	testMod = false;

}
 template <typename MACH, typename DATASET>
 MachTrainer<MACH, DATASET>::MachTrainer(DATASET & dataSet, string path):dt(dataSet){
	 threadsNum = 1;
	 machPath = path;
	 rcdFile = NULL;
	 readMachList(machPath);
	for(int i = 0; i < foldsNum; i ++){
		machList.push_back(construct(foldsList[i]));
		
	}
	predictInit();
	
}
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::operator ()(string name, float val){
	for(int i = 0; i < foldsNum; i ++){
		(*machList[i])(name, val);
	}
}
template <typename MACH, typename DATASET>
int  MachTrainer<MACH, DATASET>::trainInitialize(){
	int ret, lastIdx;
	for(int i = 0; i < foldsNum ; i++){
		machList[i]->initLoad(i == 0);
		int idx = machList[i]->getRoundIdx();
		if(i == 0 || idx < lastIdx){
			lastIdx = idx;
			ret = i;
		}
	}
	return ret;
}
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::train(){
	if(testMod){
		dt.setThreadsNum(1);
		resultRcorder[0].clear();
		resultRcorder[0].push_back(machList[0]->train());
	}else{
		int initIdx = trainInitialize();
		dt.setThreadsNum(threadsNum);
		pthread_t * tid = new pthread_t[threadsNum];
		void ** ret = new void *[threadsNum];
		for(int i = initIdx; i < (foldsNum + threadsNum -1)/threadsNum; i++){
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
	machList[0]->saveCurrentConfigText();
}
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::validate(int num){
	dt.testFold = false;
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
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::record(bool testSet){
	int num = resultRcorder[0].size();
	if(rcdFile == NULL){			
		rcdFile = new ofstream [num];
		for(int i = 0; i < num; i++){
			stringstream nm;
			nm<<machPath<<"rcd_"<<i<<".csv";
			string fl = nm.str();
			if(fileIsExist(fl)){
				rcdFile[i].open(nm.str(), ios::app);
			}else{
				rcdFile[i].open(nm.str());
			}
		}
	}	
	if(testMod){			
		for(int j = 0; j < num; j++){
			rcdFile[j]<<machList[0]->getRoundIdx()<<",";
			for(int i = 0; i < 2; i++){	
				rcdFile[j]<<resultRcorder[i][j]<<",";
			}
			rcdFile[j]<<endl;
		}
	}else{
		for(int j = 0; j < num; j++){
			rcdFile[j]<<machList[0]->getRoundIdx()<<",";
			double avg = 0;
			for(int i = 0; i < foldsNum + 1; i++){	
				if(!testSet && i == foldsNum){
					break;
				}
				rcdFile[j]<<resultRcorder[i][j]<<",";
				if(i < foldsNum){
					avg += resultRcorder[i][j];
				}
			}
			avg /= foldsNum;
			rcdFile[j]<<avg<<endl;
		}
	}

}
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::test(int num){
	dt.testFold = true;
	dt.setThreadsNum(2);
	if(testDataList.size() == 0){
		dt.setDataList(testDataList, TestDataSet, 0);
	}
	dt.loadDataList(0, testDataList);
	dt.makeBatch(0, num, false);
	if(testMod){
		machList[0]->predict(dt.Y[0], dt.Xpre[0], dt.seriesLen);	
		swap(dt.Tpre[0], dt.T[0]);
		vector<float> res = dt.getResult(0);
		resultRcorder[1].clear();
		resultRcorder[1].push_back(0);
		for(int j = 0; j < res.size(); j++){
			resultRcorder[1].push_back(res[j]);				
		}
		cout<<"\n";
		for(int j = 0; j < resultRcorder[1].size(); j++){
			cout<<resultRcorder[1][j]<<"\t";				
		}
	}else{
		for(int i = 0; i< foldsNum; i++){
			machList[i]->predict(dt.Y[1], dt.Xpre[0], dt.seriesLen);			
			for(int j = 0; j < dt.seriesLen; j++){
				if(i == 0){
					dt.Y[0][j] = dt.Y[1][j];
				}else{
					dt.Y[0][j] += dt.Y[1][j];
				}
			}
		}
		for(int j = 0; j < dt.seriesLen; j++){
			dt.Y[0][j] /= foldsNum;			
		}
		swap(dt.Tpre[0], dt.T[0]);
		vector<float> res = dt.getResult(0);
		resultRcorder[foldsNum].clear();
		resultRcorder[foldsNum].push_back(0);
		for(int j = 0; j < res.size(); j++){
			resultRcorder[foldsNum].push_back(res[j]);				
		}
		cout<<"\n";
		for(int j = 0; j < resultRcorder[foldsNum].size(); j++){
			cout<<resultRcorder[foldsNum][j]<<"\t";				
		}
	}
}
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::verify(int samplesNum, bool testSet){
	dt.roundIdx = machList[0]->getRoundIdx();
	validate(samplesNum);
	if(testSet){
		test(samplesNum);
	}
	record(testSet);
}
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::predict(){
	for(int i = 0; i < foldsNum; i++){
		machList[i]->predict(dt.forecastY, dt.forecastX, dt.forecastLen);
		for(int j = 0; j < dt.forecastLen; j++){
			if(i == 0){
				dt.forecastSumY[j] = dt.forecastY[j];
			}else{
				dt.forecastSumY[j].add(dt.forecastY[j]);
			}
		}
	}	
	for(int j = 0; j < dt.forecastLen; j++){
		dt.forecastY[j] = dt.forecastSumY[j]/foldsNum;
	}
	dt.forecastY[0].T().exportData(dt.forecastOutput);
};
template <typename MACH, typename DATASET>
void  MachTrainer<MACH, DATASET>::predictInit(){
	for(int i = 0; i < foldsNum; i++){
		machList[i]->predictInit(roundIdxList[i]);		
		//machList[i]->predictInit(MainFile);		
	
	}
	machList[0]->showConfigSetting();
}
template <typename MACH, typename DATASET>
void MachTrainer<MACH, DATASET>::readMachList(vector<int> & foldList, vector<int> & roundList, string path){
	struct _finddata_t fb;   //查找相同属性文件的存储结构体
	path = path + "machine_*";        
	long    handle = _findfirst(path.c_str(),&fb);	
	int fold;
	int round;
	if (handle != -1){		
		MACH::readMachName(fold, round, fb.name);
		foldList.push_back(fold);
		roundList.push_back(round);
		while (_findnext(handle, &fb) == 0){
			MACH::readMachName(fold, round, fb.name);
			foldList.push_back(fold);
			roundList.push_back(round);
		}
	}	
};
template <typename MACH, typename DATASET>
void MachTrainer<MACH, DATASET>::readMachList(string path){
	ifstream fl(path + "bestMachs.txt");
	foldsList.clear();
	roundIdxList.clear();
	if(!fl){
		readMachList(foldsList, roundIdxList, path);
	}else{
		int fold;
		int round;
		while(fl>>fold && fl>>round){
			foldsList.push_back(fold);
			roundIdxList.push_back(round);			
		}
		fl.close();
		
		
	}
	foldsNum = foldsList.size();
	cout<<"\n读取了"<<foldsNum<<"个算法文件";	
	int * prt = foldsList.data();
	dt("foldsNum", *max_element(prt, prt + foldsNum) + 1) ;	
};
