template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::step(){	
	do{
		forward();			
		backward();		
	}while(!Search.move(Ws, grads, loss, dt.randBatch));	
}
template <typename TYPE, bool CUDA>
void * ANNBase<TYPE, CUDA>::threadStep( void * _this){
	((ANNBase<TYPE, CUDA> *)_this)->step();
	return NULL;
}
template <typename TYPE, bool CUDA>
void * ANNBase<TYPE, CUDA>::threadMakeBatch( void * _this){	
	srand(time(NULL) + randSeeder);
	dataSetBase<TYPE, CUDA> & dt = ((ANNBase<TYPE, CUDA> *)_this)->dt;
	dt.makeBatch(((ANNBase<TYPE, CUDA> *)_this)->batchSizeControlar * dt.trainNum);
	return NULL;
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::annealControll(){
	if(!Search.notable()){
		batchSizeControlar *= batchInceaseRate;
		if(batchSizeControlar < maxBatchSize){
			batchSizeControlar = maxBatchSize;
			batchTrainRounds *= roundsDeceaseRate;
			if(batchTrainRounds<1){
				batchTrainRounds = 1;
			
			}
		}
	}
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::threadsTasks(){
	if(dt.randBatch){
		if(batchSize == 0 || (trnCount % int(batchTrainRounds) == 0)){
			loadBatch();
			Search.changeBatch();	

		}	
		annealControll();	
		pthread_t tid1, tid2;
		void *ret1,*ret2;
		pthread_create(&tid1, NULL, threadStep, (void *)this);
		if(trnCount % int(batchTrainRounds) == 0){
			randSeeder = rand();
			pthread_create(&tid2, NULL, threadMakeBatch, (void *)this);
			pthread_join(tid2, &ret2);
		}
		pthread_join(tid1,&ret1);
	}else{
		if(batchSize == 0){
			loadBatch();
			Search.changeBatch();
			batchSizeControlar = dt.trainNum;
		}
		step();
	}
}

template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::showAndRecord(){
	float tm = rcdTimer.get(false);
	cout<<"\n"<<trnCount<<"("<<Search.rounds()<<")   data loss:"<<trainLoss<<"   valid loss:"<<validLoss<<"   min_loss:"<<minValidLoss;
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
	rcdFile<<trnCount<<","<<trnLoss<<","<<vldLoss<<","<<minValidLoss<<","<<batchTrainRounds<<","<<batchSize<<","<<Search.getZ()<<","<<tm<<endl;
	dt.showResult();
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::trainInit(){	
	if(fileIsExist(binFileName())){
		load();				
	}else{
		trnCount = 0;
		rcdTimer.set();
		minValidLoss = 0;
		trainLoss = 0;
		batchTrainRounds = initBatchTrainRounds;
		batchSizeControlar = initBatchSize;
		Search.reset();
	}
	
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::stepRecord(){
	if( trnCount % showFrequence == 0){
		validLoss = getValidLoss()/dt.validInitLoss;
		if(minValidLoss == 0 || minValidLoss > validLoss){
			minValidLoss = validLoss;
			bestWs = Ws;
		}
		showAndRecord();
	}
	if(trnCount % saveFrequence == (saveFrequence - 1)){
		save();
	}
	trnCount ++;
	trainLoss = Search.getLoss()/batchInitLoss;
	return true;
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::loadBatch(){
	dt.loadBatch();
	batchSize = dt.batchSize;
	batchInitLoss = dt.batchInitLoss;
}
template <typename TYPE, bool CUDA>
bool ANNBase<TYPE, CUDA>::finish(){
	if(dt.randBatch){
		return (batchSizeControlar >= maxBatchSize && batchTrainRounds<= 1 && Search.getZ() < finishZScale);
	}else{
		return (Search.getZ() < finishZScale);
	}
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::kbPause(){	
	if(_kbhit()){
		char s=getchar();
		if(s == 'p'||s=='P'){
			dt.pauseAction();
		}
	}
}


