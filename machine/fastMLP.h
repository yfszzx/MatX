template <typename TYPE>
class fastMLP:public ANNBase<TYPE, true>, fastMat<TYPE, true>{
private:
	int samplesNum;	
	TYPE * Win;
	TYPE * Wout;
	TYPE * Bin;
	TYPE * Bout;
	TYPE * grad_out;
	TYPE * grad_in;
	TYPE * grad_Bout;
	TYPE * grad_Bin;
	TYPE * hide;
	TYPE * hideDiff;
	TYPE * Yp;
	TYPE * diff;
	int weightLen;
	void setSamplesNum(int num){
		if(num == samplesNum){
			return;
		}
		if(hide  != NULL){
			cuWrap::malloc((void **)&hide, nodes * num);
			cuWrap::malloc((void **)&hideDiff, nodes * num);
			cuWrap::malloc((void **)&diff, outputNum * num);
		}
		MatXDevice tmp(num, outputNum);
		dt.Y[0] = tmp;
		Yp = prt(dt.Y[0]);
		
	}	
	void forward(const MatXDevice & x, TYPE * y, TYPE * h, int num){
		//hide = X * Win
		cuWrap::gemm(trans(x), false, num, nodes, inputNum, prt(x), Win, h, scale(x));
		//hide += Bin
		cuWrap::matPlusRowVec(h, Bin, 1, 1, num,  nodes);
		//hide = tanh(hide);
		cuWrap::tanh(h, nodes * num);
		//Y = hide * Wout
		cuWrap::gemm(false ,false, num, outputNum, nodes, h, Wout, y, 1);
		//Y += Bout
		cuWrap::matPlusRowVec(y, Bout, 1, 1, num, outputNum);
		switch(dt.actFunc){
		case SIGMOID:
			cuWrap::sigm(y, outputNum * num);
			break;
		case TANH:
			cuWrap::tanh(y, outputNum * num);
			break;
		}

	}
protected:
	MatXDevice W;
	MatXDevice gd;	
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
			regOut = val;
			break;
		case 2:
			regIn = val;
			break;
		}
	}
	virtual void initMachine(){		
		weightLen = nodes * (inputNum + 1) + (nodes + 1) * outputNum;
		W = MatXDevice::Random(weightLen)/1000;
		Win = prt(W);
		Bin = Win + nodes * inputNum;
		Wout = Bin + nodes;
		Bout = Wout + nodes * outputNum;
		Mach<<W;
	}
	virtual void predictCore( MatXDevice * _Y, MatXDevice * _X, int len = 1) {
		int num = _X[0].rows();
		MatXDevice tmpY(num, outputNum);
		MatXDevice tmpH(num, nodes);
		forward(_X[0], prt(tmpY), prt(tmpH), num);
		_Y[0] = tmpY;
	
	}
	virtual void annTrainHead(){
		grads.clear();
		gd = MatXDevice::Zero(weightLen);
		grad_in = prt(gd);
		grad_Bin = grad_in + nodes * inputNum;
		grad_out = grad_Bin + nodes;
		grad_Bout = grad_out + nodes * outputNum;
		grads<<gd;			
	}
	virtual void forward(){
		setSamplesNum(batchSize);	
		forward(dt.X[0], Yp, hide, batchSize);
	/*	//hide = X * Win
		cuWrap::gemm(trans(dt.X[0]), false, batchSize, nodes, inputNum, prt(dt.X[0]), Win, hide, scale(dt.X[0]));
		//hide += Bin
		cuWrap::matPlusRowVec(hide, Bin, 1, 1, batchSize,  nodes);
		//hide = tanh(hide);
		cuWrap::tanh(hide, nodes * batchSize);
		//Y = hide * Wout
		cuWrap::gemm(false ,false, batchSize, outputNum, nodes, hide, Wout, Yp, 1);
		//Y += Bout
		cuWrap::matPlusRowVec(Yp, Bout, 1, 1, batchSize, outputNum);
		switch(dt.actFunc){
		case SIGMOID:
			cuWrap::sigm(Yp, outputNum * batchSize);
			break;
		case TANH:
			cuWrap::tanh(Yp, outputNum * batchSize);
			break;
		}*/
	}
	virtual void backward(){
		//diff = dt.Y[0] - dt.T[0];
		cuWrap::plus(false, trans(dt.T[0]), batchSize, outputNum, Yp, prt(dt.T[0]), diff, 1, -scale(dt.T[0]));
		//dataLoss = diff.squaredNorm()/batchSize/2;
		TYPE norm = cuWrap::norm(diff, batchSize * outputNum);
		dataLoss = norm * norm / batchSize / 2;
		loss = dataLoss;
		//Ë¥¼õÏî
		if(regIn != 0){
			norm = cuWrap::norm(Win, nodes * inputNum);
			loss += regIn * norm  * norm  /2;
		}
		if(regOut != 0){
			norm  = cuWrap::norm(Wout, nodes * outputNum);
			loss += regOut * norm  * norm  /2;
		}
		//diff = activeDerivFunc(dt.actFunc, diff, dt.Y[0]);
		if(dt.actFunc == TANH){
			cuWrap::tanhDeriv(diff, Yp, batchSize * outputNum);
		}else if(dt.actFunc == SIGMOID){
			cuWrap::sigmDeriv(diff, Yp, batchSize * outputNum);
		}

		TYPE batchInv = TYPE(1.0f)/batchSize;
		//grad_out = hide.T() * diff / batchSize;
		cuWrap::gemm(true , false, nodes, outputNum, batchSize, hide, diff, grad_out, batchInv);
		//grad_Bout = diff.sum()/batchSize;
		cuWrap::rowSum(grad_Bout, diff, batchInv, false, batchSize, outputNum);
		//hideDiff = diff * Wout.T() ;
		cuWrap::gemm(false , true, batchSize, outputNum,  nodes, diff, Wout, hideDiff, 1);

		//¼¤Àøº¯Êý
		cuWrap::tanhDeriv(hideDiff, hide, nodes * batchSize);
		//grad_in = dt.X[0].transpose() * hideDiff / batchSize;
		cuWrap::gemm(!trans(dt.X[0]) , false, inputNum,  nodes, batchSize, prt(dt.X[0]), hideDiff, grad_in, scale(dt.X[0]) * batchInv);
		//grad_Bin = hideDiff.sum()/ batchSize;
		cuWrap::rowSum(grad_Bin, hideDiff, batchInv, false, batchSize, nodes);

		//grad_in += regIn * Win;		
		if(regIn != 0){
			cuWrap::plus(false, false, inputNum, nodes, grad_in, Win, grad_in, 1, regIn);
		}
		//grad_out += regOut * Wout;
		if(regOut != 0){
			cuWrap::plus(false, false, nodes, outputNum, grad_out, Wout, grad_out, 1, regOut);
		}
	}
public:	
	fastMLP(dataSetBase<TYPE, true> & dtSet, string path):ANNBase(dtSet, path){
		initConfig();
		samplesNum = 0;
		hide  = NULL;
	};
	~fastMLP(){
		if(hide != NULL){
			cuWrap::free(hide);
			cuWrap::free(hideDiff);
			cuWrap::free(diff);
		}		
	}
};