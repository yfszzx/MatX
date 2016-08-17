
template <typename TYPE, bool CUDA>
class testMLP:public ANNBase<TYPE, CUDA>{
protected:
	MatX Win;
	MatX Wout;
	MatX Bin;
	MatX Bout;
	MatX grad_out;
	MatX grad_in;
	MatX grad_Bout;
	MatX grad_Bin;
	MatX hide;	
	int nodes;
	TYPE reg;
	virtual void annInitConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "reg", 0);
	}
	virtual void annSetConfigValue(int idx, float val){
		switch(idx){
		case 0:
			nodes = val;
			break;
		case 1:
			reg = val;
			break;
		}
	}
	virtual void initMachine(){
		Win =  MatX::Random(inputNum, nodes)/100;
		Wout =  MatX::Random(nodes, outputNum)/100;
		Bin =  MatX::Random(1, nodes)/100;
		Bout =  MatX::Random(1, outputNum)/100;
		Mach<<Win<<Wout<<Bin<<Bout;
	}
	
	virtual void annTrainHead(){
		grads<<grad_in<<grad_out<<grad_Bin<<grad_Bout;
	}
	virtual void forward(){
		hide = tanh(X[0] * Win + Bin);
		Y[0] = activeFunc(dt.actFunc, hide * Wout + Bout);					
	}
	virtual void backward(){
		MatX diff = Y[0] - T[0];
		MatX hideDerv = (TYPE)1.0f - square(hide);
		MatX tmpWout = Wout.T().replicate(Y[0].rows(), 1);
		MatX gDiff = tmpWout.cwiseProduct(hideDerv) ;
		MatX g = gDiff * Win.T();
		MatX gW = g * Win;
		MatX gDervOut =  hideDerv.cwiseProduct(gW).sum();
		MatX t = gDiff.cwiseProduct(hide).cwiseProduct(gW);
		MatX gDervBin = t.sum() * (-2);
		MatX gDervIn =  g.T() * gDiff - (X[0].T() * t) * 2;
		dataLoss = diff.squaredNorm()/batchSize/2;
		dLoss += dataLoss;
		TYPE gL = reg * g.norm2() / batchSize / 2;
		gLoss += gL;
		loss = dataLoss + gL;
		diff = activeDerivFunc(dt.actFunc, diff, Y[0]);
		grad_out = hide.T() * diff ; 
		grad_out += reg * gDervOut;
		grad_Bout = diff.sum();
		diff = diff * Wout.T() ; 
		diff = diff.cwiseProduct (hideDerv);
		grad_in = X[0].T() * diff ;
		grad_in += reg * gDervIn;
		grad_Bin = diff.sum();
		grad_Bin += reg * gDervBin;
		grads /= batchSize;

	}
public:	
	testMLP(dataSetBase<TYPE, CUDA> & dtSet, string path, int foldIdx):ANNBase(dtSet, path, foldIdx){
		initConfig();
		dLoss = 0;
		gLoss = 0;
	};
	TYPE dLoss;
	TYPE gLoss;
	virtual void predict( MatX * _Y, MatX * _X, int len = 1) {
		_Y[0] = activeFunc(dt.actFunc, tanh(_X[0] * Win + Bin) * Wout + Bout);	
		
	}
};
