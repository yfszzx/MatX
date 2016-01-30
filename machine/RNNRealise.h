template <typename TYPE, bool CUDA>
void RNN<TYPE, CUDA>::initWs(bool trainMod){
	Win =  MatriX<TYPE, CUDA>::Random(inputNum, nodes)/1000;
	Wout =  MatriX<TYPE, CUDA>::Random(nodes, outputNum)/1000;
	Bin =  MatriX<TYPE, CUDA>::Random(1, nodes)/1000;
	Bout =  MatriX<TYPE, CUDA>::Random(1, outputNum)/1000;
	Wjnt =  MatriX<TYPE, CUDA>::Random(nodes, nodes)/1000;
	Ws<<Win<<Wout<<Wjnt<<Bin<<Bout;		
	
	if(trainMod && grads.num() == 0){
		grad_out = Wout;
		grad_in = Win;
		grad_Bout = Bout;
		grad_Bin = Bin;
		grad_jnt = Wjnt;
		grads<<grad_in<<grad_out<<grad_jnt<<grad_Bin<<grad_Bout;		
	}

}
template <typename TYPE, bool CUDA>
RNN<TYPE, CUDA>::RNN(int _nodes, seriesDataBase<TYPE, CUDA> &dtSet):ANNBase(_nodes, dtSet){	
	regIn = 0;
	regOut = 0;	
	regJnt = 0;		
};
template <typename TYPE, bool CUDA>
void RNN<TYPE, CUDA>::forward(){
	stat[0] = tanh(dt.X[0] * Win + Bin);
	dt.Y[0] = activeFunc(dt.actFunc, stat[0] * Wout + Bout);
	for(int i = 1; i < dt.seriesLen; i++){
		stat[i] = tanh(dt.X[i] * Win + stat[i-1] * Wjnt + Bin);
		dt.Y[i] = activeFunc(dt.actFunc, stat[i] * Wout + Bout);
	}
}
template <typename TYPE, bool CUDA>
void RNN<TYPE, CUDA>::backward(){
	MatriX<TYPE, CUDA> diff;
	MatriX<TYPE, CUDA> diffLast =  MatriX<TYPE, CUDA>::Zero(batchSize, nodes);
	grads = 0;
	dataLoss = 0;
	for(int i = dt.seriesLen - 1; i >= dt.preLen ; i--){
		diff = dt.Y[i] - dt.T[i];
		dataLoss += diff.squaredNorm();
		diff = activeDerivFunc(dt.actFunc, diff, dt.Y[i]);	
		grad_out += stat[i].transpose() * diff ; 
		grad_Bout += diff.sum();
		diff = diff * Wout.transpose() + diffLast;
		diff = diff.cwiseProduct ( (TYPE)1.0f - square(stat[i]));
		grad_jnt += stat[i-1].transpose() * diff;
		grad_in += dt.X[i].transpose() * diff ;
		grad_Bin += diff.sum();
		diffLast = diff * Wjnt.transpose();
	}
	int num = batchSize * (dt.seriesLen - dt.preLen);
	grads /=  num;
	grad_in += regIn * Win;
	grad_out += regOut * Wout;
	grad_jnt += regJnt * Wout;
	dataLoss = dataLoss / num / 2;
	loss = dataLoss + (regIn * Win.squaredNorm() + regOut * Wout.squaredNorm() + regJnt * Wjnt.squaredNorm())/2;
}

template <typename TYPE, bool CUDA>
void RNN<TYPE, CUDA>::predict( MatriX<TYPE, CUDA> * Y,  MatriX<TYPE, CUDA>* X, int len) {
	MatriX<TYPE, CUDA> st = tanh(X[0] * Win + Bin);
	Y[0] = activeFunc(dt.actFunc, st * Wout + Bout);
	for(int i = 1; i < len; i++){
		st = tanh(X[i] * Win + st * Wjnt + Bin);
		Y[i] = activeFunc(dt.actFunc, st * Wout + Bout);
	}	
};

template <typename TYPE, bool CUDA>
void RNN<TYPE, CUDA>::setRegulars(TYPE * regVal){
	regIn = regVal[0];
	regOut = regVal[1];
	regJnt = regVal[2];
};