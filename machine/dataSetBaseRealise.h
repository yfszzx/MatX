template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeTrainAndValidList(int * &valid, int * &train){
	validNum = 0;
	for(int i = 0; i<dataNum; i++){
		if(validBoolList[i]){
			validNum ++;
		}
	}				
	train = new int[dataNum -  validNum ];
	valid = new int[ validNum ];
	int c_t = 0;
	int c_v = 0;
	for(int i = 0; i<dataNum; i++){
		if(validBoolList[i]){
			valid[c_v] = i;
			c_v++;
		}else{
			train[c_t] = i;
			c_t++;
		}
	}
	trainNum = dataNum - validNum;
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeValid( ){
	int *validList;
	makeTrainAndValidList(validList, trainList);
	validInitLoss = 0;
	for(int i = 0; i< seriesLen; i ++){
		Xv[i] = X0[i].colsMapping(validList, validNum).transpose();
		Tv[i] = T0[i].colsMapping(validList, validNum).transpose();
		validInitLoss += Tv[i].allMSE()/2;
	}		
	validInitLoss /= seriesLen;
	delete [] validList;
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::initDataSpace(){
	X0 = new  MatriX<TYPE, false>[seriesLen];
	T0 = new  MatriX<TYPE, false>[seriesLen];
	for(int i = 0; i< seriesLen; i++){
		X0[i] =  MatriX<TYPE, false>::Zero(inputNum, dataNum);
		T0[i] =  MatriX<TYPE, false>::Zero(outputNum, dataNum);
	}
	X = new  MatriX<TYPE, CUDA>[seriesLen];
	Y = new  MatriX<TYPE, CUDA>[seriesLen];
	T = new  MatriX<TYPE, CUDA>[seriesLen];
	Xv = new  MatriX<TYPE, CUDA>[seriesLen];
	Tv = new  MatriX<TYPE, CUDA>[seriesLen];
	Yv = new  MatriX<TYPE, CUDA>[seriesLen];
	Xhost = new  MatriX<TYPE, false>[seriesLen];
	Thost = new  MatriX<TYPE, false>[seriesLen];

}

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeBatch(int size){
	if(validFoldIdx == -1){
		Assert("还未设置训练集和验证集,执行makeValid(int validIdx)设置");
	}
	batchInitLoss = 0;
	if(randBatch){
		batchSize = size;
		int * batchList ;		
		batchList = new int[batchSize];
		float tm;
		for(int i = 0; i<batchSize; i++){
			batchList[i] = trainList[Mrand(trainNum)];
		}
		for(int i = 0; i< seriesLen; i++){
			Xhost[i] = X0[i].colsMapping(batchList, batchSize).T();
			Thost[i] = T0[i].colsMapping(batchList, batchSize).T();
			batchInitLoss +=  Thost[i].allMSE()/2;

		}		
		delete [] batchList;
	}else{
		batchSize = trainNum;
		for(int i = 0; i< seriesLen; i++){
			Xhost[i] = X0[i].colsMapping(trainList, batchSize).T();
			Thost[i] = T0[i].colsMapping(trainList, batchSize).T();
			batchInitLoss += Thost[i].allMSE()/2 ;
		}
	}
	batchInitLoss /= seriesLen;
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::init(int iptNum, int optNum,   activeFunctionType  _actFunc,bool rndBatch, int crossNum, int seriNum){
	inputNum = iptNum;
	outputNum = optNum;
	randBatch = rndBatch;
	seriesLen = seriNum;
	actFunc = _actFunc;
	crossFolds = crossNum;
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeValid(int validIdx){
	if(validIdx < 0 || validIdx >= crossFolds){
		Assert("验证集序号超出范围");
	}
	if(validBoolList != NULL){
		delete [] validBoolList;
		delete [] trainList;
	}
	validBoolList = new bool[dataNum];
	makeValidList(validIdx);
	makeValid();
	validFoldIdx = validIdx;
	maxCorr = -1;
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::loadBatch(){
	for(int i = 0; i< seriesLen; i++){
		X[i] = Xhost[i];
		T[i] = Thost[i];
	}
}
template <typename TYPE, bool CUDA>
dataSetBase<TYPE, CUDA>::dataSetBase(){
	trainList = NULL;
	validBoolList = NULL;
	X0 = NULL;
	dataNum = 0;
	inputNum = 0;
	outputNum = 0;
	validFoldIdx = -1;
	maxCorr = -1;
	preLen = 0;
}
template <typename TYPE, bool CUDA>
dataSetBase<TYPE, CUDA>::~dataSetBase(){
	if(validBoolList != NULL){
		delete [] validBoolList;
		delete [] trainList;
	}
	if(X0 != NULL){
		delete [] X0;
		delete [] T0;
		delete [] X;
		delete [] T;
		delete [] Y;
		delete [] Xv;
		delete [] Tv;
		delete [] Yv;
		delete [] Xhost;
		delete [] Thost;
	}
}

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::load(int dtNum){
	dataNum = dtNum;	
	initDataSpace();
	loadData();
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeValidList(int vldIdx){		
	for(int i = 0; i< dataNum ; i++){
		validBoolList[i] = (i%crossFolds == vldIdx);
	}
}

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::show(){
	cout<<"\ninputNum:"<<inputNum<<"\toutputNum:"<<outputNum<<"\trandBatch："<<randBatch;
	cout<<"\ncrossFolds:"<<crossFolds<<"\tvalidIdx:"<<validFoldIdx<<"\tdataNum:"<<dataNum<<"\ttrainNum:"<<trainNum<<"\tvalidNum"<<validNum;
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::showValidCorrel(){
	float corr;
	if(preLen == 0){
		 corr = Yv[0].correl(Tv[0]);
	}else{

	double ysum = 0; 
	double tsum = 0;
	double dot = 0;
	double ysn = 0;
	double tsn = 0;
	int n = 0;
	for(int i = preLen ; i < seriesLen; i++){
		ysum += Yv[i].allSum();
		tsum += Tv[i].allSum();
		dot += Yv[i].dot(Tv[i]);
		ysn += Yv[i].squaredNorm();
		tsn += Tv[i].squaredNorm();
		n += Yv[i].size();
	}
	corr =  (n * dot - ysum * tsum)/sqrt(( n * ysn - ysum * ysum) *( n * tsn - tsum * tsum));
	}
	
	if(maxCorr == -1 || maxCorr< corr){
		if(_finite(corr)){
			maxCorr = corr;
		}
	
	}
	cout<<"\ncorrel:"<<corr<<"\tmaxCorr"<<maxCorr;
}

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::outputValidCsv(string path, bool Continue){
	ofstream fl(path);
	for(int v = 0; v < validNum; v++){
		for(int i = preLen; i < seriesLen; i++){
			for(int o = 0; o < outputNum; o++){
				fl<<Yv[i](v, o)<<",";
			}
			fl<<",";
			for(int o = 0; o < outputNum; o++){
				fl<<Tv[i](v, o)<<",";
			}
			fl<<endl;
		}
	}
	fl.close();
	cout<<"\n已保存"<<path;
};