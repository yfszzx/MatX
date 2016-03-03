template <typename TYPE, bool CUDA>
string MachineBase< TYPE, CUDA>::binFileName(int index){	
	stringstream nm;
	if(index == AllMachine){
		nm<<machPath<<"machine_*";
		return nm.str();
	};
	if(foldIdx == NullValid){
		nm<<machPath<<"machine_NullValid_";
	}else{
		nm<<machPath<<"machine_"<<foldIdx<<"_";
	}
	if(index == MainFile){
		nm<<"MainFile.bin";
	}else{
		nm<<index<<".bin";
	}
	return nm.str();
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::saveConfig(ofstream & fl){
	int num = configRecorder.size();
	fl.write((char *)&num, sizeof(int));
	for(int i = 0; i < num; i++){
		fl.write((char *)&configRecorder[i], sizeof(float));
		fl.write((char *)configName[i].c_str(), sizeof(char) * 256);
	}
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::loadConfig(ifstream & fl){
	int num;
	fl.read((char *)&num, sizeof(int));
	float val;
	char str[256];
	for(int i = 0; i < num; i++){
		fl.read((char *)&val, sizeof(float));
		fl.read((char *)str, sizeof(char) * 256);
		(*this)(str, val);
	}	
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::save(int index){
	string file = binFileName(index);
	cout<<"\n正在保存"<<file;
	ofstream fl(file, ios::binary);
	saveConfig(fl);
	saveParameters(fl);
	fl.write((char *)&trainRoundIdx, sizeof(int));
	Mach.save(fl);	
	fl.close();	
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::load(int index){
	string file= binFileName(index);
	cout<<"\n正在读取"<<file;
	ifstream fl(file, ios::binary);
	loadConfig(fl);
	Mach.clear();
	initMachine();
	loadParameters(fl);
	fl.read((char *)&trainRoundIdx, sizeof(int));
	Mach.read(fl);	
	fl.close();
}
/*
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::clear(){
	cout<<"\n是否要删除"<<machPath<<"下的所有算法文件?(y/n)";
	string s;
	cin>>s;
	if(s[0]!='y'){
		return;
	}
	struct _finddata_t fb;   //查找相同属性文件的存储结构体
	string  path = binFileName(AllMachine, AllMachine);          
	long    handle = _findfirst(path.c_str(),&fb);
	if (handle != -1){
		path = machPath + fb.name;
		if(fileIsExist(path)){
			remove(path.c_str());
		}
		while (0 == _findnext(handle,&fb)){
			path = machPath + fb.name;
			if(fileIsExist(path)){
				remove(path.c_str());
			}
		}
	}
	path = machPath + "recorder_*.csv";
	long    handle = _findfirst(path.c_str(),&fb);
	if (handle != -1){
		path = machPath + fb.name;
		if(fileIsExist(path)){
			remove(path.c_str());
		}
		while (0 == _findnext(handle,&fb)){
			path = machPath + fb.name;
			if(fileIsExist(path)){
				remove(path.c_str());
			}
		}
	}
}*/


template <typename TYPE, bool CUDA>
MachineBase< TYPE, CUDA>::MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path, int foldIndex):dt(dtSet){
	checkFold(path);
	inputNum = dt.inputNum;
	unsuperviseDim = inputNum;
	outputNum = dt.outputNum;
	machPath = path;
	supervise = true;
	finishFlag = false;
	foldIdx = foldIndex;
	trainRoundIdx = 0;
	testMod = false;
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::initConfig(){
	initConfigValue();
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::initSet(int configIdx, string name, float val){
	while(configIdx >= configRecorder.size()){
		configRecorder.push_back(0);
		configName.push_back("");
	}
	configName[configIdx] = name;
	configRecorder[configIdx]= val;
	setConfigValue(configIdx, val);
};
template <typename TYPE, bool CUDA>
MachineBase< TYPE, CUDA>::~MachineBase(){
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::operator ()(string s, float val){
	int idx = -1;
	for(int i = 0; i<configName.size(); i++){
		if(configName[i]==s){
			idx = i;
			break;
		}
	}
	if(idx == -1){
		cout<<"\n没有找到参数"<<s;
		return;
	}
	configRecorder[idx] = val;
	setConfigValue(idx, val);
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::kbGet(string s){
	float val;
	cout<<"\n输入"<<s<<":";
	cin>>val;
	(*this)(s, val);
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::showConfigSetting(){
	for(int i = 0; i <configName.size(); i ++){
		if(configName[i].size() > 0){
			cout<<"\n["<<i<<"]"<<configName[i]<<"\t"<<configRecorder[i];
		}		
	}
};
template <typename TYPE, bool CUDA>
int MachineBase< TYPE, CUDA>::getUnsupDim(){
	return unsuperviseDim;
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::unsupervise(){
	supervise = false;
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::predictInit(){
	if(Mach.num() == 0){
		loadMach(MainFile);
		showConfigSetting();		
	}
	predictHead();
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::predict(MatriX<TYPE, CUDA> * _Y, MatriX<TYPE, CUDA>* _X, int seriesLen = 1){
/*
	if(supervise){
		MatriX<double, CUDA> * Yt = new MatriX<double, CUDA>[seriesLen];
		for(int i = 0; i< dt.crossFolds; i++){
			if(foldsMach[i].num()){
				continue;
			}
			Mach = foldsMach[i];
			predictCore(_Y, _X, seriesLen);
			if(i == 0){
				for(int k = 0; k < seriesLen; k ++){
					Yt[k] = _Y[k];
				}
			}else{
				for(int k = 0; k < seriesLen; k ++){
					Yt[k].add(_Y[k]);
				}
			}
		}
		for(int k = 0; k < seriesLen; k ++){
			_Y[k] = Yt[k]/dt.crossFolds;
		}
	}else{
		predictCore(_Y, _X, seriesLen);
	}*/
};
/*
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::showValidsResult(){	
	/ *if(supervise){
		dt.loadDatas();
		vector<MatX> rcdT;
		vector<MatX> rcdY;
		for(int i = 0; i< dt.crossFolds; i++){
			if(!fileIsExist(binFileName(i))){
				continue;
			}
			dt.makeValid(i);		
			load(false, i);	
			if(supervise){
				cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
			}else{
				cout<<"\nLoss:"<<getValidLoss();
			}
			dt.showResult();
			int num = 0;
			for(int j = dt.preLen; j<dt.seriesLen; j++){
				rcdT.push_back(dt.Tv[j]);
				rcdY.push_back(dt.Yv[j]);
				num += dt.Tv[j].size();	
			}			
			cout<<"\nsamples num:"<<num;
		}

		if(supervise){
			MatXG validT(rcdT.data(), rcdT.size());
			MatXG validY(rcdY.data(), rcdY.size());
			cout<<"\n\nWhole data set:";
			cout<<"\nLoss:"<<(validT - validY).squaredNorm()/validT.size()/validT.MSE();
			cout<<"\tsamples num:"<<validT.size();
			dt.showValidsResult(validT, validY);
		}
	}* /
};*/

template <typename TYPE, bool CUDA>
TYPE MachineBase< TYPE, CUDA>::getValidLoss(){
	/*predictCore(dt.Yv,  dt.Xv, dt.seriesLen);
	if(supervise){
		double ls = 0;
		for(int i = dt.preLen ; i < dt.seriesLen; i++){
			ls += (dt.Yv[i] - dt.Tv[i]).norm2();
		}
		return ls/dt.validNum/(dt.seriesLen - dt.preLen)/2;
	}else{
		return unsupervisedExamine(dt.Yv, dt.Xv, dt.seriesLen);		
	}*/
};


template <typename TYPE, bool CUDA>
void * MachineBase<TYPE, CUDA>::threadTrain( void * _this){	
	MachineBase<TYPE, CUDA> & mach = *(MachineBase<TYPE, CUDA> *)_this;
	srand(time(NULL) + mach.randSeeder);
	mach.trainCore();
	mach.finishFlag = mach.trainAssist();	
	return NULL;
}
template <typename TYPE, bool CUDA>
void * MachineBase<TYPE, CUDA>::threadMakeBatch( void * _this){	
	MachineBase<TYPE, CUDA> & mach = *(MachineBase<TYPE, CUDA> *)_this;
	srand(time(NULL) + mach.randSeeder);
	mach.dt.makeBatch(mach.foldIdx, mach.getBatchSize(), true);
	return NULL;
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::loadBatch(){
	batchInitLoss =  dt.loadBatch(foldIdx, X, Y, T);
	batchSize = X[0].rows();	
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::loadMach(int rndIdx){
	Mach.clear();
	initMachine();	
	if(fileIsExist(binFileName(MainFile))){
		load(rndIdx);
	}
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::trainInitialize(){
	if(Mach.num() == 0){
		loadMach(MainFile);
		showConfigSetting();
		
	}
	if(trainDataList.size() == 0){
		dt.setDataList(trainDataList, TrainDataSet, foldIdx);
	}
	stringstream rcdFl;
	rcdFl<<machPath<<"recorder_"<<foldIdx<<".csv";
	if(fileIsExist(binFileName(MainFile))){
		rcdFile.open(rcdFl.str(), ios::app);
		rcdFile<<"[continue]"<<endl;	
	}else{
		rcdTimer.set();
		rcdFile.open(rcdFl.str());
		recordFileHead();
	}
	dt.loadDataList(foldIdx, trainDataList);
	trainHead();
	trainCount = 0;	
}
template <typename TYPE, bool CUDA>
float MachineBase< TYPE, CUDA>::trainFinished(){	
	float loss = trainTail();
	trainRoundIdx ++;
	rcdFile.close();
	save(MainFile);
	if(!testMod){
		save(trainRoundIdx);	
	}
	return loss;
}

template <typename TYPE, bool CUDA>
float MachineBase<TYPE, CUDA>::train(){
	trainInitialize();		
	pthread_t tid1, tid2;
	void *ret1,*ret2;
	dt.makeBatch(foldIdx, getBatchSize(), true);	
	if(dt.randBatch){				
		do{
			trainCount ++;
			loadBatch();			
			randSeeder = rand();
			pthread_create(&tid1, NULL, threadTrain, (void *)this);			
			pthread_create(&tid2, NULL, threadMakeBatch, (void *)this);			
			pthread_join(tid1, &ret1);
			pthread_join(tid2, &ret2);
			if(finishFlag){
				break;
			}			
		}while(1);
	}else{
		loadBatch();
		do{
			trainCount ++;
			trainCore();
			if(trainAssist()){
				break;
			}
		}while(1);
	}
	return trainFinished();	
}
template <typename TYPE, bool CUDA>
vector<float> MachineBase<TYPE, CUDA>:: validate(int validNum){
	if(Mach.num() == 0){
		loadMach(MainFile);
		showConfigSetting();		
	}
	if(validDataList.size() == 0){
		dt.setDataList(validDataList, ValidDataSet, foldIdx);
	}
	dt.loadDataList(foldIdx, validDataList);
	dt.makeBatch(foldIdx, validNum, false);	
	loadBatch();			
	predict(Y, X, dt.seriesLen);
	return dt.getResult(foldIdx);
	//float l = dt.getLoss(Y, T);
	//cout<<"\nvalid loss"<<dt.getLoss(Y, T)/batchInitLoss;
	
}
