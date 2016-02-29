template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::step(){	
	do{		
		forward();			
		backward();	
	}while(!Search.move(Mach, grads, loss, dt.randBatch));	
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::annealControll(){
	if(!Search.notable()){
		batchTrainRounds *= roundsDeceaseRate;
		if(batchTrainRounds < 1){
			batchTrainRounds = 1;
			batchSizeControlar *= batchInceaseRate;
			if(batchSizeControlar > maxBatchSize){
				batchSizeControlar = maxBatchSize;			
			}
		}
	}
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::recordFileHead(){
	rcdFile<<"rounds,count,loss,batchTrainRound,batchSize,Z,time"<<endl;	
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::showAndRecord(){
	float tm = rcdTimer.get(false);
	cout<<"\n"<<trainCount<<"("<<Search.rounds()<<")   loss:"<<trainLoss<<"   valid loss:"<<validLoss;
	cout<<"\nbatch size:"<<batchSize<<"\nmean step:"<<Search.avgStep;
	if(dt.randBatch){
		cout<<"\ttrain rounds:"<<batchTrainRounds;
	}
	cout<<"\tZ:"<<Search.getZ()<<"\ttime:"<<tm;
	rcdFile<<trainCount","<<Search.rounds()<<","<<trainLoss<<","<<batchTrainRounds<<","<<batchSize<<","<<Search.getZ()<<","<<tm<<endl;
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::stepRecord(){
	if( trainCount % showFrequence == 0){
		showAndRecord();
	}
	if(trainCount % saveFrequence == (saveFrequence - 1)){
		getValidLoss();
		save();
	}
	trainLoss = Search.getLoss()/batchInitLoss;
	return true;
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::finish(){
	return (batchSizeControlar >= maxBatchSize && batchTrainRounds<= 1 && Search.getZ() < finishZScale);	
};


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
