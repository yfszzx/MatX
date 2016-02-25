template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeTrainAndValidList(int * &valid, int * &train){
	validNum = 0;
	trainNum = 0;
	vector<char> list;
	int Vnum = 0;
	for(int i = 0; i<dataNum; i++){
		if(validBoolList[i] == 1){
			Vnum ++;
		}
	}
	for(int i = 0; i<dataNum; i++){
		if(validBoolList[i] == 1){
			if(rand()%1000 < float(validBatchNum)/Vnum * 1000){
				validNum ++;
				list.push_back(true);
			}else{
				list.push_back(false);
			}
		}else if(validBoolList[i] == 0){
			trainNum ++;
		}
	}	
	train = new int[ trainNum ];
	valid = new int[ validNum ];
	int c_t = 0;
	int c_v = 0;
	int pos = 0;
	for(int i = 0; i<dataNum; i++){
		if(validBoolList[i] == 1){
			if(list[pos]){
				valid[c_v] = i;
				c_v++;
			}
			pos ++;
		}else if(validBoolList[i] == 0){
			train[c_t] = i;
			c_t++;
		}
	}
	if(validSampleList != NULL){
		delete [] validSampleList;
	}
	validSampleList = new char[list.size()];
	memcpy(validSampleList, list.data(), _msize(validSampleList));

}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeValid(){
	makeValidList(validFoldIdx);
	int *validList;
	makeTrainAndValidList(validList, trainList);
	validInitLoss = 0;
	for(int i = 0; i< seriesLen; i ++){
		Xv[i] = X0[i].colsMapping(validList, validNum).T();
		Tv[i] = T0[i].colsMapping(validList, validNum).T();
		if(i >= preLen){
			validInitLoss += Tv[i].allMSE()/2 * outputNum;
		}
	}		
	validInitLoss /= (seriesLen - preLen);
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
	/*if(validFoldIdx == -1){
		Assert("还未设置训练集和验证集,执行makeValid(int validIdx)设置");
	}*/
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
			batchInitLoss +=  Thost[i].allMSE()/2 * outputNum;

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
void dataSetBase<TYPE, CUDA>::init(int iptNum, int optNum){
	inputNum = iptNum;
	outputNum = optNum;	
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeValid(int validIdx){
	validFoldIdx = validIdx;
	if(validIdx != NullValid){
		if(validIdx < 0 || validIdx >= crossFolds){
			Assert("验证集序号超出范围");
		}
		if(validBoolList != NULL){
			delete [] validBoolList;
			delete [] trainList;
		}
		validBoolList = new char[dataNum];
	
		makeValid();	
		maxCorr = -1;
	}else{
		if(trainList != NULL){
			delete [] trainList;
		}
		trainList = new int[ dataNum ];
		for(int i = 0; i<dataNum; i++){
			trainList[i] = i;
		}
		validInitLoss = 1;
		trainNum = dataNum;
		validNum = 0;
	}
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::loadBatch(){
	for(int i = 0; i< seriesLen; i++){
		X[i] = Xhost[i];
		T[i] = Thost[i];
	}
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::loadDatas(){
	createSamples();
	if(dataNum < crossFolds){
		crossFolds = dataNum;
	}
	loatTestSamples();
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
	seriesLen = 1;
	randBatch = false;
	crossFolds = 5;
	actFunc = LINEAR;
	seriesMod = false;
	validBatchNum = 10000;
	validSampleList = NULL;	
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
	if(validSampleList != NULL){
		delete [] validSampleList;
	}
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::loadSamples(const TYPE * _X, const TYPE * _T, int _dataNum){
	if(seriesMod){
		Assert("序列数据调用loadSamples函数缺少sample的序列长度参数");
	}
	dataNum = _dataNum;
	initDataSpace();
	X0[0].importData(_X);
	T0[0].importData(_T);	
	
	for(int i = 0; i < pretreatment.size(); i++){
		MatX tmpX;
		MatX tmpY;
		tmpX = X0[0].T();	
		pretreatment[i]->predict(&tmpY, &tmpX);
		X0[0] = tmpY.T();
		inputNum = pretreatment[i]->getUnsupDim();
	}	

};
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
	cout<<"\nactFunc"<<actFunc;
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

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::set(string name, float val){
	if(name == "randBatch"){
		randBatch = val;
		return;
	}
	if(name == "crossFolds"){
			crossFolds = val;
			return;
	}

	if(name == "validNum"){
		validBatchNum = val;
		return;
	}
	if(name == "activeFunc"){
		actFunc = (activeFunctionType)((int)val);
		return;
	}	
	if(seriesMod){
		if(name == "preLen"){
			preLen = val;
			return;
		}
		if(name == "seriesLen"){
			seriesLen = val;	
			return;
		}
	}
	cout<<"\n不存在参数 "<<name;
};

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::operator()(string name, float val){
	set(name, val);
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::setPretreat(MachineBase<TYPE, CUDA> * pre){
	pretreatment.push_back(pre);
	pre->predictInit();
	
}