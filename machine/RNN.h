template <typename TYPE, bool CUDA>
class RNN:public ANNBase<TYPE, CUDA>{
protected:
	MatX Win;
	MatX Wout;
	MatX Bin;
	MatX Bout;
	MatX grad_out;
	MatX grad_in;
	MatX grad_Bout;
	MatX grad_Bin;	
	MatX Wjnt;
	MatX grad_jnt;
	MatX *hide;
	int nodes;
	float regIn;
	float regOut;
	float regJnt;
	virtual void annInitConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut", 0);
		initSet(2, "regIn", 0);
		initSet(3, "regJnt", 0);
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
		case 3:
			regJnt = val;
			break;
		}
	}
	virtual void forward(){
		hide[0] = tanh(X[0] * Win + Bin);
		Y[0] = activeFunc(dt.actFunc, hide[0] * Wout + Bout);
		for(int i = 1; i < dt.seriesLen; i++){
			hide[i] = tanh(X[i] * Win + hide[i-1] * Wjnt + Bin);
			Y[i] = activeFunc(dt.actFunc, hide[i] * Wout + Bout);
		}
	};
	virtual void backward(){
		MatX diff;
		MatX diffLast =  MatX::Zero(batchSize, nodes);
		grads = 0;
		dataLoss = 0;
		for(int i = dt.seriesLen - 1; i >= dt.preLen ; i--){
			diff = Y[i] - T[i];
			dataLoss += diff.squaredNorm();
			diff = activeDerivFunc(dt.actFunc, diff, Y[i]);	
			grad_out += hide[i].transpose() * diff ; 
			grad_Bout += diff.sum();
			diff = diff * Wout.transpose() + diffLast;
			diff = diff.cwiseProduct ( (TYPE)1.0f - square(hide[i]));
			grad_jnt += hide[i-1].transpose() * diff;
			grad_in += X[i].transpose() * diff ;
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
	virtual void  initMachine(){
		Win =  MatX::Random(inputNum, nodes)/1000;
		Wout =  MatX::Random(nodes, outputNum)/1000;
		Bin =  MatX::Random(1, nodes)/1000;
		Bout =  MatX::Random(1, outputNum)/1000;
		Wjnt =  MatX::Random(nodes, nodes)/1000;
		Mach<<Win<<Wout<<Wjnt<<Bin<<Bout;	
		
	}

	virtual void annTrainHead(){
		if(hide == NULL){
			
			hide = new MatX[dt.seriesLen];
		}
		grad_out = Wout;
			grad_in = Win;
			grad_Bout = Bout;
			grad_Bin = Bin;
			grad_jnt = Wjnt;
			//grads<<grad_in<<grad_out<<grad_jnt<<grad_Bin<<grad_Bout;
					grads<<grad_in<<grad_out<<grad_jnt<<grad_Bin<<grad_Bout;
	}
public:	
	RNN(seriesDataBase<TYPE, CUDA> &dtSet, string path, int foldIdx):ANNBase(dtSet, path, foldIdx){
	 initConfig();
	 hide = NULL;
	};
	~RNN(){
		if(hide != NULL){
			delete [] hide;
		}
	}
	virtual void predict( MatX * _Y,  MatX* _X, int len = 1) {
		MatX st = tanh(_X[0] * Win + Bin);
		_Y[0] = activeFunc(dt.actFunc, st * Wout + Bout);
		for(int i = 1; i < len; i++){
			st = tanh(_X[i] * Win + st * Wjnt + Bin);
			_Y[i] = activeFunc(dt.actFunc, st * Wout + Bout);
		}	
	}
};

