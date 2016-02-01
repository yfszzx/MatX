template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::saveParameters(ofstream & fl){
	fl.write((char *)&batchTrainRounds, sizeof(float));
	fl.write((char *)&batchSizeControlar, sizeof(float));
	fl.write((char *)& minValidLoss, sizeof(float));
	fl.write((char *)& trainCount, sizeof(int));
	double tm = rcdTimer.get();
	fl.write((char *)& tm, sizeof(double));
	int rn = Search.rounds();
	fl.write((char *)& rn, sizeof(int));
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::loadParameters(ifstream & fl){
	fl.read((char *)&batchTrainRounds, sizeof(float));
	fl.read((char *)&batchSizeControlar, sizeof(float));
	fl.read((char *)& minValidLoss, sizeof(float));
	fl.read((char *)& trainCount, sizeof(int));
	double tm;
	fl.read((char *)& tm, sizeof(double));
	rcdTimer.set(tm);
	int rn;
	fl.read((char *)& rn, sizeof(int));
	Search.setRounds(rn);
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::recordFileHead(){
	rcdFile<<"rounds,data_loss,valid_loss,min_loss,batchTrainRound,batchSize,Z,time"<<endl;	
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::initWs(bool trainMod){
	if(trainMod){
		minValidLoss = 0;
		trainLoss = 0;
		batchTrainRounds = initBatchTrainRounds;
		batchSizeControlar = initBatchSize;
		Search.reset();
	}	
	annInitWs(trainMod);
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::setConfigValue(int idx, float val){
	configRecorder[idx] = val;
	if(idx < subConfigsNum || subConfigsNum == -1){
		annSetConfigValue(idx , val);
		Search.changeBatch();
		return;
	}
	idx -= subConfigsNum;
	switch(idx){
	case 0:
		Search.debug = (int)val;
		break;
	case 1:
		Search.setAlg((int)val);
		break;
	case 2:
		Search.lr = val;
		break;
	case 3:
		Search.momentum = val;
		break;
	case 4:
		Search.setConfirmRounds((int)val);
		break;
	case 5:
		Search.Zscale = val;
		break;
	case 6:
		initBatchTrainRounds = val;
		break;
	case 7:
		roundsDeceaseRate = val;
		break;
	case 8:
		batchInceaseRate = val;
		break;
	case 9:
		saveFrequence = val;
		break;
	case 10:
		showFrequence = val;
		break;
	case 11:
		initBatchSize = val;
		break;
	case 12:
		maxBatchSize = val;
		break;
	case 13:
		finishZScale = val;
		break;
	}
	
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::initConfigValue(){
	subConfigsNum = -1;
	annInitConfigValue();
	for(int i = 0; i < 100; i++){
		if(configName[i].size()>0){
			if(subConfigsNum < i){
				subConfigsNum = i;
			}
		}
	}
	subConfigsNum ++;
	initSet(subConfigsNum, "debug", false);
	initSet(subConfigsNum + 1, "alg", 3);
	initSet(subConfigsNum + 2, "lr", 0.5);
	initSet(subConfigsNum + 3, "momentum", 0.5);
	initSet(subConfigsNum + 4, "confirmRounds", 50);
	initSet(subConfigsNum + 5, "Zscale", 1);
	initSet(subConfigsNum + 6, "initBatchTrainRounds", 4);
	initSet(subConfigsNum + 7,  "roundsDeceaseRate", 0.8);
	initSet(subConfigsNum + 8, "batchInceaseRate", 1.2);
	initSet(subConfigsNum + 9, "saveFreq", 100);
	initSet(subConfigsNum + 10, "showFreq",10);
	initSet(subConfigsNum + 11, "initBatchSize", 0.1);
	initSet(subConfigsNum + 12, "maxBatchSize", 0.8);
	initSet(subConfigsNum + 13, "finishZScale", -0.5);	

};
template <typename TYPE, bool CUDA>
ANNBase<TYPE, CUDA>::ANNBase(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){}
template <typename TYPE, bool CUDA>
int ANNBase<TYPE, CUDA>::getBatchSize(){
	return batchSizeControlar * dt.trainNum;
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::train(){		
	int cnt = 0;
	do{	
		step();				
		cnt ++;
		if(!dt.randBatch){
			stepRecord();	
			kbPause();	
			trainCount ++;
			if(Search.getZ() < finishZScale){
				break;
			}
		}else{
			annealControll();	
		}
	}while(!dt.randBatch || cnt <= batchTrainRounds);
	
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::trainOperate(){
	kbPause();	
	stepRecord();		
	
	Search.changeBatch();	
	if(finish()){
		return true;
	}
	return false;

}