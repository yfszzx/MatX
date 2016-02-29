template <typename TYPE, bool CUDA>
int dataSetBase<TYPE, CUDA>::thrdIdx(int foldIdx){
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
void dataSetBase<TYPE, CUDA>::makeData(int foldIdx, int * list, int num){	
	int idx = thrdIdx(foldIdx);
	dataSize[idx] = num;
	initLoss[idx] = 0;	
	for(int i = 0; i< seriesLen; i ++){
		Xpre[idx][i] = X0[i].colsMapping(list, num).T();		
		Tpre[idx][i] = T0[i].colsMapping(list, num).T();
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
void dataSetBase<TYPE, CUDA>::makeBatch(int foldIdx, int num){	
	int idx = thrdIdx(foldIdx);
	if(randBatch){
		vector<int> list;
		for(int i = 0; i < num; i++){
			if(float(rand()%10000) / 10000 * subDataNum[idx] < num){
				list.push_back(i);				
			}			
		}
		makeData(foldIdx, list.data(), list.size());

	}else{
		makeData(foldIdx, dataList[idx], subDataNum[idx]);
	}	
}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::initDataSpace(){
	if(pretreatment.size() == 0){
		X0 = new  MatriX<TYPE, false>[seriesLen];
		T0 = new  MatriX<TYPE, false>[seriesLen];
		for(int i = 0; i< seriesLen; i++){
			X0[i] =  MatriX<TYPE, false>::Zero(inputNum, dataNum);
			T0[i] =  MatriX<TYPE, false>::Zero(outputNum, dataNum);
		}
		X = new  MatX * [threadsNum];
		Y = new  MatX * [threadsNum];
		T = new  MatX * [threadsNum];
		dataSize = new int[threadsNum];
		initLoss = new double[threadsNum];
		dataList = new int *[threadsNum];
		subDataNum = new int[threadsNum];
		Xpre = new  MatX * [threadsNum];
		Tpre = new  MatX * [threadsNum];

		for(int i = 0; i < threadsNum; i++){
			dataList[i] = NULL;
			X[i] = new MatX[seriesLen];
			Y[i] = new MatX[seriesLen];
			T[i] = new MatX[seriesLen];
			Xpre[i] = new MatX[seriesLen];
			Tpre[i] = new MatX[seriesLen];
		}
		
	}

}
template <typename TYPE, bool CUDA>
void dataSetBase<TYPE, CUDA>::init(int iptNum, int optNum, int actf = LINEAR, bool seri = false){
	inputNum = iptNum;
	outputNum = optNum;	
	actFunc = (activeFunctionType)actf;
	seriesMod = seri;
}
template <typename TYPE, bool CUDA>
dataSetBase<TYPE, CUDA>::dataSetBase(){
	X0 = NULL;
	dataNum = 0;
	threadsNum = 1;
	preLen = 0;
	seriesLen = 1;
	randBatch = false;
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

		delete [] X0;
		delete [] T0;
		delete [] dataSize;
		delete [] initLoss;
		
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
		threadsNum = val;
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
void dataSetBase<TYPE, CUDA>::setDataList(vector<int> & list, char mod, int foldIdx){
	list.clear();
	switch(mod){
	case TrainDataSet:
		for(int i = 0; i < dataNum - testNum; i++){
			if(i % foldsNum != foldIdx){
				list.push_back(i);			
			}
		}
		break;
	case ValidDataSet:
		for(int i = 0; i < dataNum - testNum; i++){
			if(i % foldsNum == foldIdx){
				list.push_back(i);			
			}
		}
		break;
	case TestDataSet:
		for(int i = dataNum - testNum; i < dataNum; i++){
				list.push_back(i);			
		}
		break;
	case PredictDataSet:
		for(int i = 0; i< dataNum; i++){
			list.push_back(i);		
		}
		break;
	}
};
template <typename TYPE, bool CUDA>
float dataSetBase<TYPE, CUDA>:: getLoss(MatX * y, MatX * t){
	double loss = 0;
	for(int i = preLen; i< seriesLen; i ++){
		loss += (y[i] - t[i]).square().allSum();
	}
	loss /= (seriesLen - preLen) * t[0].rows() * 2;
	return  loss * outputNum;
}	