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
	//TYPE * Yp;
	TYPE * diff;
	int weightLen;
	void setSamplesNum(int num){
		if(num == samplesNum){
			return;
		}
		samplesNum = num;
		if(hide != NULL){
			cuWrap::free(hide);
			cuWrap::free(hideDiff);
			cuWrap::free(diff);
		}
		cuWrap::malloc((void **)&hide, nodes * num * sizeof(TYPE));
		cuWrap::malloc((void **)&hideDiff, nodes * num * sizeof(TYPE));
		cuWrap::malloc((void **)&diff, outputNum * num * sizeof(TYPE));
		//MatXDevice tmp(num, outputNum);
		//dt.Y[0] = tmp;
		//Yp = prt(dt.Y[0]);		
	}	
	void forward(const MatXDevice & x, TYPE * y, TYPE * h){
		TYPE sclW = 1;//scale(W);
		int num = x.rows();
		//hide = X * Win
		cuWrap::gemm(trans(x), false, num, nodes, inputNum, prt(x), Win, h, scale(x) * sclW);
		//hide += Bin
		cuWrap::matPlusRowVec(h, Bin, 1, sclW, num,  nodes);
		//hide = tanh(hide);
		cuWrap::tanh(h, nodes * num);
		//Y = hide * Wout
		cuWrap::gemm(false ,false, num, outputNum, nodes, h, Wout, y, sclW);
		//Y += Bout
		cuWrap::matPlusRowVec(y, Bout, 1, sclW, num, outputNum);
		switch(dt.actFunc){
		case SIGMOID:
			cuWrap::sigm(y, outputNum * num);
			break;
		case TANH:
			cuWrap::tanh(y, outputNum * num);
			break;
		}
		//检验
		/*MatXDevice tWin (inputNum, nodes);
		MatXDevice tBin (1, nodes);
		MatXDevice tWout(nodes, outputNum);
		MatXDevice tBout(1, outputNum);
		MatXDevice Out(num, outputNum);
		Out.importData(y);
		tWin.importData(Win);
		tWout.importData(Wout);
		tBout.importData(Bout);
		tBin.importData(Bin);
		th = tanh(x * tWin + tBin);
	//	Dbg2(sclW, (Out -  activeFunc(dt.actFunc, th * tWout + tBout)).norm());*/
	}
	void setW(){
		Win = prt(W);
		Bin = Win + nodes * inputNum;
		Wout = Bin + nodes;
		Bout = Wout + nodes * outputNum;
	}
	void setGrad(){
		grad_in = prt(gd);
		grad_Bin = grad_in + nodes * inputNum;
		grad_out = grad_Bin + nodes;
		grad_Bout = grad_out + nodes * outputNum;
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
	/*MatXDevice th;*/
	virtual void initMachine(){		
		weightLen = nodes * (inputNum + 1) + (nodes + 1) * outputNum;
		W = MatXDevice::Random(weightLen) / 100;
		cuWrap::malloc((void **)&Win, sizeof(TYPE) * weightLen	);
		Bin = Win + nodes * inputNum;
		Wout = Bin + nodes;
		Bout = Wout + nodes * outputNum;
		W.exportData(Win, true);
		cuWrap::malloc((void **)&grad_in, sizeof(TYPE) * weightLen	);
		grad_Bin = grad_in + nodes * inputNum;
		grad_out = grad_Bin + nodes;
		grad_Bout = grad_out + nodes * outputNum;
		Mach<<W;
	}
	virtual void predictCore( MatXDevice * _Y, MatXDevice * _X, int len = 1) {
		//setW();
		W.exportData(Win, true);
		int num = _X[0].rows();
		MatXDevice tmpY(num, outputNum);
		MatXDevice tmpH(num, nodes);		
		forward(_X[0], prt(tmpY), prt(tmpH));
		_Y[0] = tmpY;
	}
	virtual void annTrainHead(){
		grads.clear();
		gd = MatXDevice::Zero(weightLen);
		grads<<gd;			
	}
	virtual void forward(){
	//	setW();
		setSamplesNum(batchSize);	
		W.exportData(Win, true);
		dt.Y[0] = MatXDevice::Zero(batchSize, outputNum);
		forward(dt.X[0], prt(dt.Y[0]), hide);
	}
	virtual void backward(){

		TYPE sclW = 1;//scale(W);
		//setGrad();
		//diff = dt.Y[0] - dt.T[0];
		cuWrap::plus(false, trans(dt.T[0]), batchSize, outputNum, prt(dt.Y[0]), prt(dt.T[0]), diff, 1, - scale(dt.T[0]));
			/*MatXDevice tDiff(batchSize, outputNum);
			tDiff.importData(diff);
			//Dbg((dt.Y[0] - dt.T[0] - tDiff).norm()/tDiff.norm());*/
		//dataLoss = diff.squaredNorm()/batchSize/2;
		TYPE norm = cuWrap::norm(diff, batchSize * outputNum);
		dataLoss = norm * norm / batchSize / 2;
		loss = dataLoss;
		//衰减项
		if(regIn != 0){
			norm = cuWrap::norm(Win, nodes * inputNum) * sclW;
			loss += regIn * norm  * norm  /2;
		}
		if(regOut != 0){
			norm  = cuWrap::norm(Wout, nodes * outputNum) * sclW;
			loss += regOut * norm  * norm  /2;
		}
		//diff = activeDerivFunc(dt.actFunc, diff, dt.Y[0]);
		if(dt.actFunc == TANH){
			cuWrap::tanhDeriv(diff, prt(dt.Y[0]), batchSize * outputNum);
		}else if(dt.actFunc == SIGMOID){
			cuWrap::sigmDeriv(diff, prt(dt.Y[0]), batchSize * outputNum);
		}

		TYPE batchInv = TYPE(1.0f)/batchSize;

		
	
		//grad_out = hide.T() * diff / batchSize;
		cuWrap::gemm(true , false, nodes, outputNum, batchSize, hide, diff, grad_out, batchInv);
			/*MatXDevice thh(batchSize, nodes);
			thh.importData(hide);
			MatXDevice tg(nodes, outputNum);
			tg.importData(grad_out);*/
			//Dbg((th - thh).norm()/ (thh).norm());
			//Dbg((tg * batchSize - th.T()*tDiff).norm()/ (th.T()*tDiff).norm());

		
		//grad_Bout = diff.sum()/batchSize;
		cuWrap::rowSum(grad_Bout, diff, batchInv, false, batchSize, outputNum);
			/*MatXDevice tgB(1, outputNum);
			tgB.importData(grad_Bout);		*/
			//Dbg((tgB * batchSize - tDiff.sum()).norm()/tDiff.sum().norm() );
		
		
		//hideDiff = diff * Wout.T() ;
		cuWrap::gemm(false , true, batchSize, nodes, outputNum, diff, Wout, hideDiff, sclW);
			/*MatXDevice thd(batchSize, nodes);
			thd.importData(hideDiff);
			MatXDevice tWout(nodes, outputNum);
			tWout.importData(Wout);*/
			//Dbg((tDiff * tWout.T() - thd).norm()/thd.norm());
	
		//激励函数导数
		cuWrap::tanhDeriv(hideDiff, hide, nodes * batchSize);
			/*MatXDevice thd2(batchSize, nodes);
			thd2.importData(hideDiff);*/
			//Dbg((thd2 - thd.cwiseProduct ( (TYPE)1.0f - square(th))).norm()/thd2.norm());

		//grad_in = dt.X[0].T() * hideDiff / batchSize;			
		cuWrap::gemm(!trans(dt.X[0]) , false, inputNum,  nodes, batchSize, prt(dt.X[0]), hideDiff, grad_in, scale(dt.X[0]) * batchInv);
			/*MatXDevice tg2(inputNum, nodes);
			tg2.importData(grad_in);*/
			//Dbg((tg2 * batchSize - dt.X[0].T() * thd2 ).norm()/(tg2.norm() * batchSize));

		//grad_Bin = hideDiff.sum()/ batchSize;
		cuWrap::rowSum(grad_Bin, hideDiff, batchInv, false, batchSize, nodes);
			/*MatXDevice tbg2(1, nodes);
			tbg2.importData(grad_Bin);*/
			//Dbg((tbg2 * batchSize - thd2.sum()).norm()/(tbg2.norm() * batchSize));
		//grad_in += regIn * Win;		
		if(regIn != 0){
			cuWrap::plus(false, false, inputNum, nodes, grad_in, Win, grad_in, 1, regIn);
		}
		//grad_out += regOut * Wout;
		if(regOut != 0){
			cuWrap::plus(false, false, nodes, outputNum, grad_out, Wout, grad_out, 1, regOut);
		}

		gd.importData(grad_in, true);
	/*	MatXDevice tWin (inputNum, nodes);
		MatXDevice tBin (1, nodes);
		MatXDevice two(nodes, outputNum);
		MatXDevice tBout(1, outputNum);
		tWin.importData(Win);
		two.importData(Wout);
		tBout.importData(Bout);
		tBin.importData(Bin);
		MatXDevice tdf = dt.Y[0] - dt.T[0];
		tdf = activeDerivFunc(dt.actFunc, tdf, dt.Y[0]);
		MatXDevice tgo = th.T() * tdf ; 
		MatXDevice tgbo = tdf.sum();
		tdf = tdf * two.T() ; 
		tdf = tdf.cwiseProduct ( (TYPE)1.0f - square(th));
		MatXDevice tgi = dt.X[0].T() * tdf ;
		MatXDevice tgbi =  tdf.sum();
		double sm1 = 0;
		double sm2 = 0;
		for(int i = 0; i< tgi.size(); i++){
			TYPE aa = gd[i] * batchSize;
			TYPE bb = tgi[i];
			sm1 += (aa - bb )*(aa - bb);
			sm2 += aa * aa;
		}
		Dbg2(sm1, sm2);*/
		//gd /= batchSize;
		//Dbg((tgbi - tbg2* batchSize).norm()/tgbi.norm());
		//Dbg((tgi - tg2 * batchSize).norm()/tgi.norm());
		//Dbg((tgbo - tgB * batchSize).norm()/tgbo.norm());
		//Dbg((tgo - tg * batchSize).norm()/tgo.norm());
	

		/*// 梯度检验 
		double sm1 = 0;
		double sm2 = 0;
		double sm3 = 0;
		float scl = 0.0001;
		for(int i = 0; i < grads.num(); i++){
			for(int j = 0; j < grads[i].size(); j++){

				float tmp = Mach[i][j];
				tmp += scl;
				Mach[i].assignment(j,tmp);
				float d = grads[i][j];
				forward(dt.X[0], Yp, hide);
				float result = (dt.Y[0] - dt.T[0]).squaredNorm()/batchSize/2;
				sm1 += (result - loss)/scl * (result - loss)/scl;
				sm2 += d * d;
				sm3 += ((result - loss)/scl - d) *  ((result - loss)/scl - d);
				tmp -= scl;
				Mach[i].assignment(j,tmp);
				if(j == nodes * inputNum - 1){
					Dbg3(sm1, sm2, sm3);	
				}
				if(j == nodes * inputNum + nodes - 1){
					Dbg3(sm1, sm2, sm3);	
				}
				if(j == nodes * inputNum + nodes + nodes * outputNum - 1){
					Dbg3(sm1, sm2, sm3);	
				}
			}

		}
		Dbg3(sm1, sm2, sm3);		*/
		
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