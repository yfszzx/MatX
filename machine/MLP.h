template <typename TYPE, bool CUDA>
class MLP:public ANNBase<TYPE, CUDA>{
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
	TYPE regIn;
	TYPE regOut;
	virtual void annInitConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut", 0);
		initSet(2, "regIn", 0);
	}
	virtual void annSetConfigValue(int idx, float val){
		switch(idx){
		case 0:
			nodes = val;
			break;
		case 1:
			regIn = val;
			break;
		case 2:
			regOut = val;
			break;
		}
	}
	virtual void initWs(bool trainMod = true){
		Win =  MatX::Random(inputNum, nodes)/1000;
		Wout =  MatX::Random(nodes, outputNum)/1000;
		Bin =  MatX::Random(1, nodes)/1000;
		Bout =  MatX::Random(1, outputNum)/1000;
		Ws<<Win<<Wout<<Bin<<Bout;
		grads.clear();
		grads<<grad_in<<grad_out<<grad_Bin<<grad_Bout;			
	}
	virtual void predict( MatX * _Y, MatX * _X, int len = 1) {
		_Y[0] = activeFunc(dt.actFunc, tanh(_X[0] * Win + Bin) * Wout + Bout);	
	}
	virtual void forward(){
		hide = tanh(dt.X[0] * Win + Bin);
		dt.Y[0] = activeFunc(dt.actFunc, hide * Wout + Bout);	
	}
	virtual void backward(){
		MatX diff = dt.Y[0] - dt.T[0];
		dataLoss = diff.squaredNorm()/batchSize/2;
		loss = dataLoss + (regIn * Win.squaredNorm() + regOut * Wout.squaredNorm())/2;
		diff = activeDerivFunc(dt.actFunc, diff, dt.Y[0]);
		grad_out = hide.transpose() * diff ; 
		grad_Bout = diff.sum();
		diff = diff * Wout.transpose() ; 
		diff = diff.cwiseProduct ( (TYPE)1.0f - square(hide));
		grad_in = dt.X[0].transpose() * diff ;
		grad_Bin = diff.sum();
		grads /= batchSize;
		grad_in += regIn * Win;
		grad_out += regOut * Wout;
	}
public:	
	MLP(dataSetBase<TYPE, CUDA> & dtSet, string path):ANNBase(dtSet, path){
		initConfig();
	};
};