template <typename TYPE, bool CUDA>
string MachineBase< TYPE, CUDA>::binFileName(int idx){	
	stringstream nm;
	if(idx == NullValid){
		nm<<machPath<<"machine_NullValid.bin";
	}else if(idx == AllMachine){
		nm<<machPath<<"machine_*";
	}else{
		nm<<machPath<<"machine_"<<idx<<".bin";
	}
	return nm.str();
}
template <typename TYPE, bool CUDA>
bool MachineBase< TYPE, CUDA>::overedFile(int foldIdx){
	string binFl = binFileName(foldIdx);
	bool over = fileIsExist(binFl);
	if(over){
		bool finished;
		ifstream fl(binFl);
		fl.read((char *)&finished, sizeof(bool));
		if(!finished){
			over = false;
		}else{
			cout<<"\n"<<binFl<<"已完成训练";
		}
		fl.close();
	}
	return over;
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
void MachineBase< TYPE, CUDA>::save(bool finished){
	string file = binFileName(dt.validFoldIdx);
	cout<<"\n正在保存"<<file;
	ofstream fl(file, ios::binary);
	fl.write((char *)&finished, sizeof(bool));
	saveConfig(fl);
	saveParameters(fl);
	bestMach.save(fl);
	Mach.save(fl);	
	fl.close();
	if(finished){
		cout<<"\n"<<file<<"已完成训练";
	}
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::load(bool trainMod, int idx){
	string file= binFileName(idx);
	cout<<"\n正在读取"<<file;
	ifstream fl(file, ios::binary);
	bool finished;
	fl.read((char *)&finished, sizeof(bool));
	loadConfig(fl);
	Mach.clear();
	initMachine();
	loadParameters(fl);
	if(trainMod){
		bestMach.read(fl);
		Mach.read(fl);	

	}else{
		Mach.read(fl);
	//Mach.read(fl);
	}
	fl.close();
}

template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::showValidsResult(){	
	if(supervise){
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
	}
};

template <typename TYPE, bool CUDA>
TYPE MachineBase< TYPE, CUDA>::getValidLoss(){
	predictCore(dt.Yv,  dt.Xv, dt.seriesLen);
	if(supervise){
		double ls = 0;
		for(int i = dt.preLen ; i < dt.seriesLen; i++){
			ls += (dt.Yv[i] - dt.Tv[i]).norm2();
		}
		return ls/dt.validNum/(dt.seriesLen - dt.preLen)/2;
	}else{
		return unsupervisedExamine(dt.Yv, dt.Xv, dt.seriesLen);		
	}
};
template <typename TYPE, bool CUDA>
MachineBase< TYPE, CUDA>::MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path):dt(dtSet){
	foldsMach = NULL;
	inputNum = dt.inputNum;
	unsuperviseDim = inputNum;
	outputNum = dt.outputNum;
	machPath = path;
	supervise = true;
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
void MachineBase< TYPE, CUDA>::setBestMach(){
	bestMach = Mach;
}
template <typename TYPE, bool CUDA>
MachineBase< TYPE, CUDA>::~MachineBase(){
	if(foldsMach != NULL){
		delete [] foldsMach;
	}
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::predictInit(){
	if(foldsMach != NULL){
		delete [] foldsMach;
	}
	if(supervise){
		foldsMach = new MatGroup<TYPE ,CUDA>[dt.crossFolds];
		for(int i = 0; i< dt.crossFolds; i++){
			if(!fileIsExist(binFileName(i))){
				continue;
			}
			load(false, i);
			foldsMach[i] = Mach;
		}
	}else{
		if(!fileIsExist(binFileName(NullValid))){
			Assert("非监督算法还未训练完毕");
		}
		load(false, NullValid);
	}
	predictHead();
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::predict(MatriX<TYPE, CUDA> * _Y, MatriX<TYPE, CUDA>* _X, int seriesLen = 1){
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
	}
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
void * MachineBase<TYPE, CUDA>::threadTrain( void * _this){
	srand(time(NULL) + ((MachineBase<TYPE, CUDA> *)_this)->randSeeder);
	MachineBase<TYPE, CUDA> & t = * (MachineBase<TYPE, CUDA> *)_this;
	t.trainCore();
	t.batchFinished = t.trainAssist();	
	return NULL;
}
template <typename TYPE, bool CUDA>
void * MachineBase<TYPE, CUDA>::threadMakeBatch( void * _this){	
	dataSetBase<TYPE, CUDA> & dt = ((ANNBase<TYPE, CUDA> *)_this)->dt;
	srand(time(NULL) + ((MachineBase<TYPE, CUDA> *)_this)->randSeeder);
	dt.makeBatch(((MachineBase<TYPE, CUDA> *)_this)->getBatchSize());
	return NULL;
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::kbPause(){	
	if(_kbhit()){
		char s = getchar();
		if(s == 'p'||s=='P'){
			dt.pauseAction(this);
		}
	}
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::loadBatch(){
	dt.loadBatch();
	batchInitLoss = dt.batchInitLoss;
	batchSize = dt.batchSize;
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::trainInitialize(){
	cout<<"\n训练进行中.....";
	Mach.clear();
	initMachine();
	bestMach = Mach;
	string rcdFl = machPath + "recorder.csv";
	if(fileIsExist(binFileName(dt.validFoldIdx))){
		load(true, dt.validFoldIdx);
		showConfigSetting();
		rcdFile.open(rcdFl, ios::app);
		rcdFile<<"[continue]"<<endl;	
	}else{
		rcdTimer.set();
		if( dt.validFoldIdx == 0){
			rcdFile.open(rcdFl);
			recordFileHead();
		}else{
			rcdFile.open(rcdFl, ios::app);
		}
		rcdFile<<"[valid "<<dt.validFoldIdx<<"]"<<endl;
	}

}
template <typename TYPE, bool CUDA>
int MachineBase< TYPE, CUDA>::getUnsupDim(){
	return unsuperviseDim;
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::clear(){
		cout<<"\n是否要删除"<<machPath<<"下的所有算法文件?(y/n)";
		string s;
		cin>>s;
		if(s[0]!='y'){
			return;
		}
		struct _finddata_t fb;   //查找相同属性文件的存储结构体
		string  path = binFileName(AllMachine);          
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
		path = machPath + "recorder.csv";
		if(fileIsExist(path)){
			remove(path.c_str());
		}
		
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::trainRun(){
	checkFold(machPath);
	dt.loadDatas();
	for(int i = 0; i< dt.crossFolds; i++){			
		trainMain(i);
	}
	if(!supervise){
		trainMain(NullValid);
	}
}

template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::trainMain(int idx){
	if(overedFile(idx)){
		return;
	}
	dt.makeValid(idx);
	dt.show();

	trainInitialize();		
	trainCount = 1;	
	trainHead();
	if(dt.randBatch){
		dt.makeBatch(getBatchSize());			
		do{
			loadBatch();
			pthread_t tid1, tid2;
			void *ret1,*ret2;
			pthread_create(&tid1, NULL, threadTrain, (void *)this);
			randSeeder = rand();
			pthread_create(&tid2, NULL, threadMakeBatch, (void *)this);			
			pthread_join(tid1, &ret1);
			pthread_join(tid2, &ret2);
			if(batchFinished){
				break;
			}
			trainCount ++;
			kbPause();		
		}while(1);

	}else{
		dt.makeBatch();
		loadBatch();
		trainCore();
		trainAssist();
	}
	trainTail();
	save(true);
	rcdFile.close();
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::unsupervise(){
	supervise = false;
}