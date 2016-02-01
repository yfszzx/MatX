template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::step(){	
	do{		
		forward();			
		backward();			
	}while(!Search.move(Ws, grads, loss, dt.randBatch));	
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::annealControll(){
	if(!Search.notable()){
		batchSizeControlar *= batchInceaseRate;
		if(batchSizeControlar > maxBatchSize){
			batchSizeControlar = maxBatchSize;
			batchTrainRounds *= roundsDeceaseRate;
			if(batchTrainRounds<1){
				batchTrainRounds = 1;
			
			}
		}
	}
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::showAndRecord(){
	float tm = rcdTimer.get(false);
	cout<<"\n"<<trainCount<<"("<<Search.rounds()<<")   data loss:"<<trainLoss<<"   valid loss:"<<validLoss<<"   min_loss:"<<minValidLoss;
	cout<<"\nbatch size:"<<batchSize;
	if(dt.randBatch){
		cout<<"\ttrain rounds:"<<batchTrainRounds;
	}
	cout<<"\tZ:"<<Search.getZ()<<"\ttime:"<<tm;
	float trnLoss = trainLoss;
	float vldLoss = validLoss;
	if(trnLoss > 1){
		trnLoss = 1;
	}
	if(vldLoss > 1){
		vldLoss = 1;
	}
	rcdFile<<trainCount<<","<<trnLoss<<","<<vldLoss<<","<<minValidLoss<<","<<batchTrainRounds<<","<<batchSize<<","<<Search.getZ()<<","<<tm<<endl;
	dt.showResult();
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::stepRecord(){
	if( trainCount % showFrequence == 0){
		validLoss = getValidLoss()/dt.validInitLoss;
		if(minValidLoss == 0 || minValidLoss > validLoss){
			minValidLoss = validLoss;
			bestWs = Ws;
		}
		showAndRecord();
	}
	if(trainCount % saveFrequence == (saveFrequence - 1)){
		save();
	}
	trainLoss = Search.getLoss()/batchInitLoss;
	return true;
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::finish(){
	return (batchSizeControlar >= maxBatchSize && batchTrainRounds<= 1 && Search.getZ() < finishZScale);	
};



