template <typename TYPE, bool CUDA>
int dataSetBase<TYPE, CUDA>::thrdIdx(int foldIdx){
	if(foldIdx == NullValid){
		return 0;
	}
	return foldIdx % threadsNum;
}
template <typename TYPE, bool CUDA>
float dataSetBase<TYPE, CUDA>::loadBatch(int foldIdx, MatX * & x, MatX * & y, MatX * & t){
	int idx = thrdIdx(foldIdx);
	swap(Xpre[idx], X[idx]);
	swap(Tpre[idx], T[idx]);
	x = X[idx];
	y = Y[idx];
	t = T[idx];
	return initLoss[idx];
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::pretreat(MatX &x){
	MatX ret;
	for(int i = 0; i < pretreatment.size(); i++){
		pretreatment[i]->predict(&ret, &x);	
		x = ret;
	}	
	
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeData(int foldIdx){	
	int idx = thrdIdx(foldIdx);
	initLoss[idx] = 0;	
	for(int i = 0; i< seriesLen; i ++){
		Xpre[idx][i] = X0[i].colsMapping(batchList[idx], dataSize[idx]).T();		
		Tpre[idx][i] = T0[i].colsMapping(batchList[idx], dataSize[idx]).T();
		if(pretreatment.size() > 0){
			pretreat(Xpre[idx][i]);
		}
		if(i >= preLen){
			initLoss[idx] += Tpre[idx][i].allMSE()/2 * outputNum;
		}
	}		
	initLoss[idx] /= (seriesLen - preLen);
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::loadDataList(int foldIdx,  vector<int> & list){	
	int idx = thrdIdx(foldIdx);
	subDataNum[idx] = list.size();
	dataList[idx] = list.data();		
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::makeBatch(int foldIdx, int num, bool replacement){	//replacement 有放回的
	int idx = thrdIdx(foldIdx);
	if(randBatch){
		vector<int> list;
		int size = subDataNum[idx];
		if(replacement){			
			for(int i = 0; i < num; i++){
				list.push_back(dataList[idx][Mrand(size)]);						
			}
		}else{
			for(int i = 0; i < size; i++){
				if(float(rand()%10000) / 10000 * size < num){
					list.push_back(dataList[idx][i]);				
				}			
			}
		}
		if(dataSize[idx] != list.size()){
			if(batchList[idx] != NULL){
				delete [] batchList[idx];
			}
			dataSize[idx] = list.size();
			batchList[idx] = new int[dataSize[idx]];
		}
		memcpy(batchList[idx], list.data(), sizeof(int) * dataSize[idx]);
		

	}else{
		if(dataSize[idx] != subDataNum[idx]){
			if(batchList[idx] != NULL){
				delete [] batchList[idx];
			}
			dataSize[idx] = subDataNum[idx];
			batchList[idx] = new int[dataSize[idx]];
		}
		memcpy(batchList[idx], dataList[idx], sizeof(int) * dataSize[idx]);
	}	
	makeData(foldIdx);
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::setThreadsNum(int num){	
	if(X != NULL){
		for(int i = 0; i < threadsNum; i++){
			delete [] X[i];
			delete [] Y[i];
			delete [] T[i];
			delete [] Xpre[i];
			delete [] Tpre[i];
			if(batchList[i] != NULL){
				delete [] batchList[i];
			}
		}
		delete [] X;
		delete [] Y;
		delete [] T;
		delete [] Xpre;
		delete [] Tpre;
		delete [] dataSize;
		delete [] initLoss;
		delete [] batchList;
		

	}
	threadsNum = num;
	X = new  MatX * [threadsNum];
	Y = new  MatX * [threadsNum];
	T = new  MatX * [threadsNum];
	dataSize = new int[threadsNum];
	initLoss = new double[threadsNum];
	dataList = new int *[threadsNum];
	subDataNum = new int[threadsNum];
	Xpre = new  MatX * [threadsNum];
	Tpre = new  MatX * [threadsNum];
	batchList = new int *[threadsNum];

	for(int i = 0; i < threadsNum; i++){
		dataList[i] = NULL;
		X[i] = new MatX[seriesLen];
		Y[i] = new MatX[seriesLen];
		T[i] = new MatX[seriesLen];
		Xpre[i] = new MatX[seriesLen];
		Tpre[i] = new MatX[seriesLen];
		dataSize[i] = 0;
		batchList[i] = NULL;
	}
	
};
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::initDataSpace(){
	if(X0 == NULL){
		X0 = new  MatriX<TYPE, false>[seriesLen];
		T0 = new  MatriX<TYPE, false>[seriesLen];
		for(int i = 0; i< seriesLen; i++){
			X0[i] =  MatriX<TYPE, false>::Zero(inputNum, dataNum);
			T0[i] =  MatriX<TYPE, false>::Zero(outputNum, dataNum);
		}		
		setThreadsNum(threadsNum);
	}

}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::init(int iptNum, int optNum, int actf){
	inputNum = iptNum;
	outputNum = optNum;	
	actFunc = (activeFunctionType)actf;
}
template <typename TYPE, bool CUDA>
dataSetBase<TYPE, CUDA>::dataSetBase(){
	X0 = NULL;	
	preLen = 0;
	seriesLen = 1;
	randBatch = false;
	foldsNum = 1;
	testNum = 0;
	threadsNum = 1;
	X = NULL;
	batchList = NULL;
}
template <typename TYPE, bool CUDA>
dataSetBase<TYPE, CUDA>::~dataSetBase(){
	if(X0 != NULL){
		for(int i = 0; i < threadsNum; i++){
			delete [] X[i];
			delete [] Y[i];
			delete [] T[i];
			delete [] Xpre[i];
			delete [] Tpre[i];
		}
		delete [] X;
		delete [] Y;
		delete [] T;
		delete [] Xpre;
		delete [] Tpre;
		delete [] dataSize;
		delete [] initLoss;

		delete [] X0;
		delete [] T0;
		
		
	}	
}

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::operator()(string name, float val){
	if(name == "randBatch"){
		randBatch = val;
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
	if(name == "threadsNum"){
		setThreadsNum(val);
		return;
	}
	if(name == "foldsNum"){
		foldsNum = val;
		return;
	}
	if(name == "testNum"){
		testNum = val;
		return;
	}
	cout<<"\n不存在参数 "<<name;
}

template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::loadDataSet(const TYPE * _X, const TYPE * _T, int _dataNum){
	if(seriesMod){
		Assert("序列数据调用loadSamples函数缺少sample的序列长度参数");
	}
	dataNum = _dataNum;
	initDataSpace();
	X0[0].importData(_X);
	T0[0].importData(_T);
};
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::setPretreat(MachineBase<TYPE, CUDA> * pre){
	pretreatment.push_back(pre);
	pre->predictInit();
	inputNum = pre->getUnsupDim();
}
template <typename TYPE, bool CUDA>
bool dataSetBase<TYPE, CUDA>::trainSample(int idx, int foldIdx){
	return (idx < (dataNum - testNum)) && (idx % foldsNum != foldIdx);	
}
template <typename TYPE, bool CUDA>
bool dataSetBase<TYPE, CUDA>::validSample(int idx, int foldIdx){
	return (idx < (dataNum - testNum)) && (idx % foldsNum == foldIdx);	
}
template <typename TYPE, bool CUDA>
bool dataSetBase<TYPE, CUDA>::testSample(int idx,  int foldIdx){
	return idx >= (dataNum - testNum);
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::setDataList(vector<int> & list, int mod, int foldIdx){
	list.clear();
	if(foldIdx == NullValid){
		for(int i = 0; i < dataNum; i++){
			list.push_back(i);					
		}
		return;
	}
	
	switch(mod){
	case TrainDataSet:
		for(int i = 0; i < dataNum; i++){
			if(trainSample(i, foldIdx)){
				list.push_back(i);			
			}
		}
		cout<<"\ndataNum"<<dataNum<<"\ttrainSetNum"<<list.size();
		break;
	case ValidDataSet:
		for(int i = 0; i < dataNum; i++){
			if(validSample(i, foldIdx)){
				list.push_back(i);			
			}
		}
		cout<<"\ndataNum"<<dataNum<<"\tvalidSetNum"<<list.size();
		break;
	case TestDataSet:
		for(int i = 0; i < dataNum; i++){
			if(testSample(i, foldIdx)){
				list.push_back(i);			
			}
		}
		cout<<"\ndataNum"<<dataNum<<"\ttestSetNum"<<list.size();
		break;
	}
};
template <typename TYPE, bool CUDA>
float dataSetBase<TYPE, CUDA>:: getLoss(int foldIdx){
	int idx = thrdIdx(foldIdx);
	double loss = 0;
	for(int i = preLen; i< seriesLen; i ++){
		loss += (Y[idx][i] - T[idx][i]).square().allSum();
	}
	loss /= (seriesLen - preLen) * T[idx][0].rows() * 2;
	return  loss * outputNum/initLoss[idx];
}	
template <typename TYPE, bool CUDA>
float dataSetBase<TYPE, CUDA>:: getCorrel(int foldIdx){
	int idx = thrdIdx(foldIdx);
	double loss = 0;
	MatXG t;
	MatXG y;
	for(int i = preLen; i< seriesLen; i ++){
		t<<T[idx][i];
		y<<Y[idx][i];
	}
	return t.correl(y);	
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::show(){
	cout<<"\ninputNum:"<<inputNum<<"\toutputNum:"<<outputNum<<"\tactFunc"<<actFunc;
	cout<<"\nrandBatch："<<randBatch<<"\tfoldsNum:"<<foldsNum<<"\tdataNum:"<<dataNum<<"\ttestNum"<<testNum;	
}
template <typename TYPE, bool CUDA>
vector<float> dataSetBase<TYPE, CUDA>::getResult(int foldIdx){
	vector<float> ret;
	ret.push_back(getLoss(foldIdx));
	ret.push_back(getCorrel(foldIdx));
	getMoreResult(ret, foldIdx);
	return ret;
}
template <typename TYPE, bool CUDA>
int dataSetBase<TYPE, CUDA>::getDataNum(int idx){
	return subDataNum[thrdIdx(idx)];

}