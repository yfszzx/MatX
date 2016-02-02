template <typename TYPE, bool CUDA>
string MachineBase< TYPE, CUDA>::binFileName(int idx){
	if(idx == -1){
		idx = dt.validFoldIdx;
	}
	stringstream nm;
	nm<<machPath<<"machine_"<<idx<<".bin";
	return nm.str();
}


template <typename TYPE, bool CUDA>
bool MachineBase< TYPE, CUDA>::overedFile(){
	string binFl = binFileName();
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
void MachineBase< TYPE, CUDA>::save(bool finished){
	string file= binFileName();
	cout<<"\n正在保存"<<file;
	ofstream fl(file, ios::binary);
	fl.write((char *)&finished, sizeof(bool));
	saveParameters(fl);
	bestWs.save(fl);
	Ws.save(fl);	
	fl.close();
	if(finished){
		cout<<"\n"<<file<<"已完成训练";
	}
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::load(bool trainMod){
	string file= binFileName();
	cout<<"\n正在读取"<<file;
	ifstream fl(file, ios::binary);
	bool finished;
	fl.read((char *)&finished, sizeof(bool));
	loadParameters(fl);
	if(trainMod){
		bestWs.read(fl);
	}
	Ws.read(fl);	
	fl.close();
}

template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::wholeValidsResult(){	
	vector<MatX> rcdT;
	vector<MatX> rcdY;
	for(int i = 0; i< dt.crossFolds; i++){
		dt.makeValid(i);
		if(!fileIsExist(binFileName())){
			continue;
		}
		initWs();
		load(false);		
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();
		int num = 0;
		for(int j = dt.preLen; j<dt.seriesLen; j++){
			rcdT.push_back(dt.Tv[j]);
			rcdY.push_back(dt.Yv[j]);
			num += dt.Tv[j].size();		
		}			
		cout<<"\tsamples num:"<<num;
	}
	MatXG validT(rcdT.data(), rcdT.size());
	MatXG validY(rcdY.data(), rcdY.size());
	cout<<"\n\nWhole data set:";
	cout<<"\nLoss:"<<(validT - validY).squaredNorm()/validT.size()/validT.MSE();
	cout<<"\tsamples num:"<<validT.size();

	dt.showValidsResult(validT, validY);
};

template <typename TYPE, bool CUDA>
TYPE MachineBase< TYPE, CUDA>::getValidLoss(){
	predict( dt.Yv,  dt.Xv, dt.seriesLen);
	double ls = 0;
	for(int i = dt.preLen ; i < dt.seriesLen; i++){
		ls += (dt.Yv[i] - dt.Tv[i]).squaredNorm();
	}
	return ls/dt.validNum/(dt.seriesLen - dt.preLen)/2;
};
template <typename TYPE, bool CUDA>
MachineBase< TYPE, CUDA>::MachineBase(dataSetBase<TYPE, CUDA> & dtSet, string path):dt(dtSet){
	WSs = NULL;
	inputNum = dt.inputNum;
	outputNum = dt.outputNum;
	machPath = path;
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::initConfig(){
	initConfigValue();
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::initSet(int configIdx, string name, float val){
	configName[configIdx] = name;
	configRecorder[configIdx]= val;
	setConfigValue(configIdx, val);

};
template <typename TYPE, bool CUDA>
MachineBase< TYPE, CUDA>::~MachineBase(){
	if(WSs != NULL){
		delete [] WSs;
	}
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::initPredict(){
	if(WSs != NULL){
		delete [] WSs;
	}
	WSs = new MatGroup<TYPE ,CUDA>[dt.crossFolds];
	for(int i = 0; i< dt.crossFolds; i++){
		if(!fileIsExist(binFileName(i))){
			continue;
		}
		initWs();
		load(false);
		WSs[i] = Ws;
	}
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::Predict(MatriX<TYPE, CUDA> * _Y, MatriX<TYPE, CUDA>* _X, int seriesLen = 1){
	for(int i = 0; i< dt.crossFolds; i++){
		if(WSs[i].num()){
			continue;
		}
		predict(Y, X);
		Ws = WSs[i];
		initWs();
		load(false);
		WSs[i] = Ws;
	}
};
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::operator ()(string s, float val){
	int idx = -1;
	for(int i = 0; i<100; i++){
		if(configName[i]==s){
			idx = i;
			break;
		}
	}
	if(idx == -1){
		cout<<"\n没有找到参数"<<s;
		return;
	}
	setConfigValue(idx, val);
};

template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::showConfigSetting(){
	for(int i = 0; i < 100; i ++){
		if(configName[i].size() > 0){
			cout<<"\n["<<i<<"]"<<configName[i]<<"\t"<<configRecorder[i];
		}		
	}
};
template <typename TYPE, bool CUDA>
void * MachineBase<TYPE, CUDA>::threadTrain( void * _this){
	MachineBase<TYPE, CUDA> & t = * (MachineBase<TYPE, CUDA> *)_this;
	t.train();
	t.batchFinished = t.trainOperate();	
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
			dt.pauseAction();
		}
	}
}

template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::trainInitialize(){
	cout<<"\n训练进行中.....";
	Ws.clear();
	initWs();
	bestWs = Ws;
	string rcdFl = machPath + "recorder.csv";
	if(fileIsExist(binFileName())){
		load();
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
void MachineBase< TYPE, CUDA>::trainRun(){
	for(int i = 0; i< dt.crossFolds; i++){
		dt.makeValid(i);		
		if(overedFile()){
			continue;
		}
		trainInitialize();	
		trainMain();
		save(true);
		rcdFile.close();
	}
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::loadBatch(){
	dt.loadBatch();
	batchInitLoss = dt.batchInitLoss;
	batchSize = dt.batchSize;
}
template <typename TYPE, bool CUDA>
void MachineBase<TYPE, CUDA>::trainMain(){
	trainCount = 0;	
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
		train();
		trainOperate();
	}
}