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
		batchTrainRounds *= roundsDeceaseRate;
		if(batchTrainRounds<1){
			batchTrainRounds = 1;
			//cout<<"\n"<<batchSizeControlar<<"\t"<<maxBatchSize;
			if(batchSizeControlar < maxBatchSize){
				batchSizeControlar *= batchInceaseRate;
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
ANNBase<TYPE, CUDA>::ANNBase( int _nodes, dataSetBase<TYPE, CUDA> & dtSet):MachineBase<TYPE, CUDA>(dtSet){
	nodes = _nodes;
	Search.debug = false;
	Search.setAlg(2);
	Search.setConfirmRounds(100);
	Search.lr = 0.5;
	Search.momentum = 0.8;
	Search.Zscale = 0;
	batchTrainRounds = 8;
	roundsDeceaseRate = 0.9;
	batchInceaseRate = 1.2;
	batchSize = 0;
	maxBatchSize = 1;
	showFrequence = 10;
	saveFrequence = 100;
	finishZScale = -0.5f;
	stat = new MatriX<TYPE, CUDA>[dt.seriesLen];
}
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::trainSet(int name, float val){
	switch(name){
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
		Search.setConfirmRounds((int)val);
		break;

	case 4:
		Search.momentum = val;
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
	//cout<<"\n"<<trnCount<<"("<<Search.rounds()<<")\tdata loss:"<<trainLoss <<"\t"<< dt.batchInitLoss <<"\t"<< dataLoss<<"\t"<< batchSize;
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
ANNBase<TYPE, CUDA>::~ANNBase(){
	delete []  stat;
};
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::saveParameters(ofstream & fl){
	fl.write((char *)&batchTrainRounds, sizeof(float));
	fl.write((char *)&batchSizeControlar, sizeof(float));
	fl.write((char *)& minValidLoss, sizeof(float));
	fl.write((char *)& trnCount, sizeof(int));
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
	fl.read((char *)& trnCount, sizeof(int));
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
template <typename TYPE, bool CUDA>
void ANNBase<TYPE, CUDA>::train(){	
	trainInit();
	dt.makeBatch(batchSizeControlar * dt.trainNum);
	do{		
		threadsTasks();	
		kbPause();		
		if(!stepRecord()){
			break;
		}
		if(finish()){
			break;
		}
	}while(1);

}