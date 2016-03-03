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
void ANNBase<TYPE, CUDA>::showAndRecord(){
	float tm = rcdTimer.get(false);
	cout<<"\n"<<trainCount<<"("<<Search.rounds()<<")   data loss:"<<trainLoss;
	cout<<"\nbatch size:"<<batchSize<<"\navgStep:"<<Search.avgStep;
	if(dt.randBatch){
		cout<<"\ttrain rounds:"<<batchTrainRounds;
	}
	rcdFile<<trainCount<<","<<trainLoss<<","<<batchTrainRounds<<","<<batchSize<<","<<tm<<endl;
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::stepRecord(){
	if( trainCount % showFrequence == 0){
		showAndRecord();
	}
	meanLoss += Search.getLoss()/batchInitLoss;
	return true;
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::finish(){
	return (batchSizeControlar >= maxBatchSize && batchTrainRounds<= 1 && Search.getZ() < finishZScale);	
};