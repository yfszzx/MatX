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
void ANNBase<TYPE, CUDA>::trainHead(){
		minValidLoss = 0;
		trainLoss = 0;
		batchTrainRounds = initBatchTrainRounds;
		batchSizeControlar = initBatchSize;
		Search.reset();	
		grads.clear();
		meanLoss = 0;
		annTrainHead();
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::setConfigValue(int idx, float val){
	if(idx < subConfigsNum || subConfigsNum == -1){
		annSetConfigValue(idx , val);
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
	case 14:
		trainRounds = val;
		break;
	}
	
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::initConfigValue(){
	annInitConfigValue();
	subConfigsNum = configRecorder.size();
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
	initSet(subConfigsNum + 14, "trainRounds", 100);	

};
template <typename TYPE, bool CUDA>
ANNBase<TYPE, CUDA>::ANNBase(dataSetBase<TYPE, CUDA> & dtSet, string path, int foldIdx):MachineBase<TYPE, CUDA>(dtSet, path, foldIdx){
	subConfigsNum = -1;
	
}

template <typename TYPE, bool CUDA>
int ANNBase<TYPE, CUDA>::getBatchSize(){
	return batchSizeControlar * dt.getDataNum(foldIdx);
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::breakFlag(int cnt){

	 if(cnt < int(batchTrainRounds)){
		return false;
	}
	 if(cnt >= batchTrainRounds){
		 return true;
	 }
	 return (rand() % 1000 ) > (batchTrainRounds - int(batchTrainRounds)) * 1000;
	 
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::trainCore(){		
	int cnt = 0;
	do{	
		step();		
		cnt ++;
		if(!dt.randBatch){
			stepRecord();	
		//	kbPause();	
			trainCount ++;
			if(Search.getZ() < finishZScale){
				break;
			}
		}else{
			annealControll();	
		}		
		
		
	}while(!dt.randBatch || !breakFlag(cnt));
	
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::trainAssist(){
	stepRecord();			
	Search.changeBatch();	
	return !( trainCount < trainRounds);
	return false;

}
template <typename TYPE, bool CUDA>
float ANNBase<TYPE, CUDA>::trainTail(){	
	return meanLoss/trainRounds;
}