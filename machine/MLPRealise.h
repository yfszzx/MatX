template <typename TYPE, bool CUDA>
void MLP<TYPE, CUDA>::initWs(bool trainMod){
	Win =  MatriX<TYPE, CUDA>::Random(inputNum, nodes)/1000;
	Wout =  MatriX<TYPE, CUDA>::Random(nodes, outputNum)/1000;
	Bin =  MatriX<TYPE, CUDA>::Random(1, nodes)/1000;
	Bout =  MatriX<TYPE, CUDA>::Random(1, outputNum)/1000;
	Ws<<Win<<Wout<<Bin<<Bout;

	if(trainMod && grads.num() == 0){
		grad_out = Wout;
		grad_in = Win;
		grad_Bout = Bout;
		grad_Bin = Bin;
		grads<<grad_in<<grad_out<<grad_Bin<<grad_Bout;	
	}
}
template <typename TYPE, bool CUDA>
MLP<TYPE, CUDA>::MLP(int _nodes, dataSetBase<TYPE, CUDA> &dtSet):ANNBase(_nodes, dtSet){	
	regIn = 0;
	regOut = 0;	
};
template <typename TYPE, bool CUDA>
void MLP<TYPE, CUDA>::forward(){
	stat[0] = tanh(dt.X[0] * Win + Bin);
	dt.Y[0] = activeFunc(dt.actFunc, stat[0] * Wout + Bout);	
}
template <typename TYPE, bool CUDA>
void MLP<TYPE, CUDA>::backward(){
	MatriX<TYPE, CUDA> diff = dt.Y[0] - dt.T[0];
	dataLoss = diff.squaredNorm()/batchSize/2;
	loss = dataLoss + (regIn * Win.squaredNorm() + regOut * Wout.squaredNorm())/2;
	diff = activeDerivFunc(dt.actFunc, diff, dt.Y[0]);
	grad_out = stat[0].transpose() * diff ; 
	grad_Bout = diff.sum();
	diff = diff * Wout.transpose() ; 
	diff = diff.cwiseProduct ( (TYPE)1.0f - square(stat[0]));
	grad_in = dt.X[0].transpose() * diff ;
	grad_Bin = diff.sum();
	grads /= batchSize;
	grad_in += regIn * Win;
	grad_out += regOut * Wout;
}

template <typename TYPE, bool CUDA>
void MLP<TYPE, CUDA>::predict( MatriX<TYPE, CUDA> * Y,  MatriX<TYPE, CUDA>* X, int len) {
	Y[0] = activeFunc(dt.actFunc, tanh(X[0] * Win + Bin) * Wout + Bout);	
};
template <typename TYPE, bool CUDA>
void MLP<TYPE, CUDA>::setRegulars(TYPE * regVal){
	regIn = regVal[0];
	regOut = regVal[1];
};