template <typename TYPE, bool CUDA>
string MachineBase< TYPE, CUDA>::binFileName(int idx){
	if(idx == -1){
		idx = dt.validFoldIdx;
	}
	stringstream nm;
	nm<<path<<"machine_"<<idx<<".bin";
	return nm.str();
}
template <typename TYPE, bool CUDA>
void MachineBase< TYPE, CUDA>::trainInit(){
	cout<<"\n训练进行中.....";
	Ws.clear();
	initWs();
	bestWs = Ws;
	string rcdFl = path + "recorder.csv";
	if(fileIsExist(binFileName())){
		rcdFile.open(rcdFl, ios::app);
		rcdFile<<"[continue]"<<endl;	
	}else{
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
void MachineBase< TYPE, CUDA>::trainRun(string _path){
	path = _path;
	for(int i = 0; i< dt.crossFolds; i++){
		dt.makeValid(i);
		if(overedFile()){
			continue;
		}
		trainInit();		
		train();
		save(true);
		rcdFile.close();
	}
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
void MachineBase< TYPE, CUDA>::wholeValidsResult(string _path){
	path = _path;
	MatGroup<TYPE, CUDA> validT;
	MatGroup<TYPE, CUDA> validY;
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
			validT<<dt.Tv[j];
			validY<<dt.Yv[j];
			num += dt.Tv[j].size();		
		}			
		cout<<"\tsamples num:"<<num;
	}
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
MachineBase< TYPE, CUDA>::MachineBase(dataSetBase<TYPE, CUDA> & dtSet):dt(dtSet){
	WSs = NULL;
	inputNum = dt.inputNum;
	outputNum = dt.outputNum;
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