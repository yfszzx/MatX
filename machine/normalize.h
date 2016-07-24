template <typename TYPE, bool CUDA>
class normalize:public MachineBase<TYPE, CUDA>{	
private:	
	MatX means;
	MatX dev;
	MatXD meansD;
	MatXD devD;
	MatXD meansTmp;
	MatXD devTmp;
	int trainRounds;
	float batchNum;
	bool showResult;
	bool trainMod;
protected:
	virtual void initConfigValue(){
		initSet(0, "trainRounds",50);
		initSet(1, "batchNum",1000);	
		initSet(2, "showResult",0);	
	};
	virtual void setConfigValue(int idx, float val){
		switch(idx){
		case 0:
			trainRounds = val;
			break;
		case 1:
			batchNum= val;
			break;
		case 2:
			showResult = val;
			break;
		}

	};
	virtual void initMachine(){
		Mach<<means<<dev;
	};
	virtual void predict( MatX * _Y,  MatX* _X, int len = 1){	
		for(int i = 0; i< len; i++){
			_Y[i] = (_X[i] - means) * MatX::Diagonal(dev.cwiseInverse());
		}

	};
	virtual int getBatchSize(){
		return batchNum;
	}
	virtual void trainCore(){
		int num = (dt.seriesLen - dt.preLen) * X[0].rows();
		meansTmp = X[dt.preLen].sum();
		for(int i = dt.preLen + 1; i<dt.seriesLen; i++){
			meansTmp.add(X[i].sum());
		}
		meansTmp /= num;
		means = meansTmp;


		MatX * dtCpy = new MatX[dt.seriesLen - dt.preLen];
		dtCpy[0] = X[dt.preLen] - means;
		devTmp = square(dtCpy[0]).sum();
		for(int i =1; i< dt.seriesLen - dt.preLen; i++){
			dtCpy[i] = X[i + dt.preLen] - means;
			devTmp.add(square(dtCpy[i]).sum());
		}
		devTmp = sqrt(devTmp/num);
		delete [] dtCpy;
	}
	virtual bool trainAssist(){
		if(trainCount == 1){
			meansD = meansTmp;
			devD = devTmp;			
		}else{
			meansD += meansTmp;
			devD += devTmp;			
		}
		cout<<" round:"<<trainCount<<"\r";
		return !( trainCount < trainRounds);
	};
	virtual float trainTail(){
		means = meansD/trainRounds;
		dev = devD/trainRounds;
		if(showResult){
			showParams();
		}
		return 0;
	}
public:
	void showParams(){
		cout<<"\nmean:";
		cout<<means;
		getchar();
		cout<<"\ndev:";
		cout<<dev;
		getchar();
		
	}
	normalize(dataSetBase<TYPE, CUDA> & dtSet, string path, int validId):MachineBase<TYPE, CUDA>(dtSet, path, validId){
		initConfig();
		unsupervise();	
	}

};
