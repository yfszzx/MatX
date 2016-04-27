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
void MachineBase< TYPE, CUDA>::readMachName(int &fold, int &round, string _name){
	vector<string> name;
	split(_name, "_", name);
	if(name[1] == "NullValid"){
		fold = NullValid;
	}else{
		fold = atoi(name[1].c_str());
	}
	if(name[2] == "MainFile"){
		round = MainFile;
	}else{
		round = atoi(name[2].c_str());
	}
	
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
void MachineBase< TYPE, CUDA>::saveConfigText(string path){
	ofstream fl(path);
	int num = configRecorder.size();
	for(int i = 0; i < num; i++){
		fl<<configName[i]<<" "<<configRecorder[i]<<endl;
	}
	fl.close();
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::loadConfigText(){
	ifstream fl(machPath + "currentConfig.txt");
	int num = configRecorder.size();
	for(int i = 0; i < num; i++){
		string s;
		float v;
		fl>>s>>v;
		if(s != configName[i]){
			Assert("配置文件错误,文件参数名" + s + " 与算法参数名 " + configName[i] + " 不一致");
		}
		(*this)(s, v);
	}
	fl.close();
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
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::createConfigFile(){
	string path = machPath + "config.txt";
	if(!fileIsExist(path)){
		saveConfigText(path);
		cout<<"\n在下面这个文件中配置算法参数:\n"<<machPath << "config.txt";
		cout<<"\n[输入任意字符继续]";
		string s;
		cin>>s;
		
	}	
	string cpath = machPath + "currentConfig.txt";
	if(!fileIsExist(cpath)){
		copyFile(path, cpath);
	}
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::saveCurrentConfigText(){
	saveConfigText(machPath + "currentConfig.txt");
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
	cout<<"\n";
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
void MachineBase< TYPE, CUDA>::predictInit(int roundIdx){
	if(Mach.num() == 0){
		loadMach(roundIdx);
	}
};
template <typename TYPE, bool CUDA>
int MachineBase< TYPE, CUDA>::getRoundIdx(){
	return trainRoundIdx;
};
template <typename TYPE, bool CUDA>
TYPE MachineBase< TYPE, CUDA>::getLoss(MatX * _Y,  MatX * _T){
	double ls = 0;
	for(int i = dt.preLen ; i < dt.seriesLen; i++){
		ls += (_Y[i] - _T[i]).norm2();
	}
	return ls/_T[0].rows()/(dt.seriesLen - dt.preLen)/2;	
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
	if(fileIsExist(binFileName(rndIdx))){
		load(rndIdx);	
		loadConfigText();
	}else{
		loadConfigText();
		Mach.clear();
		initMachine();	
	}
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::initLoad(bool showConf = false){
	if(Mach.num() == 0){		
		createConfigFile();
		loadMach(MainFile);	
		if(showConf){
			showConfigSetting();	
		}
	}
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::trainInitialize(){	
	initLoad();
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
template <typename TYPE, bool CUDA>
float MachineBase<TYPE, CUDA>:: get(string name){
	int num = configRecorder.size();
	for(int i = 0; i < num; i++){
		if(name == configName[i]){
			return configRecorder[i];
		}
	}
	Assert("不存在参数" + name);
	return 0;
}
template <typename TYPE, bool CUDA>
bool MachineBase<TYPE, CUDA>:: isTrained(){
	return fileIsExist(binFileName(MainFile));
}
