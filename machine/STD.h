template <typename TYPE, bool CUDA>
class STD:public MachineBase<TYPE, CUDA>{	
private:	
	MatX means;
	MatX var;
	
	MatXD meansD;
	MatXD varD;
	MatXD meansTmp;
	MatXD varTmp;
	int trainRounds;
	float batchScale;
protected:
	virtual void initConfigValue(){
		initSet(0, "trainRounds",50);
		initSet(1, "batchScale",0.1);		
	};
	virtual void setConfigValue(int idx, float val){
		switch(idx){
		case 0:
			trainRounds = val;
			break;
		case 1:
			batchScale = val;
			break;
		}

	};
	virtual void initMachine(){
		Mach<<means<<var;
	};
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){
		MatX scl = MatX::Diagonal(var);
		for(int i = 0; i< len; i++){
			_Y[i] = (_X[i] - means) * scl;
		}
	};
	virtual int getBatchSize(){
		return dt.trainNum  * batchScale;
	}
	virtual void trainHead(){};
	virtual void trainCore(){
		int num = (dt.seriesLen - dt.preLen) * dt.X[0].rows();

		meansTmp = dt.X[dt.preLen].sum();
		for(int i = dt.preLen + 1; i<dt.seriesLen; i++){
			meansTmp.add(dt.X[i].sum());
		}
		meansTmp /= num;
		means = meansTmp;

		varTmp = square(dt.X[dt.preLen] - means).sum();
		for(int i =1; i< dt.seriesLen - dt.preLen; i++){
			varTmp.add(square( dt.X[i + dt.preLen] - means).sum());
		}
		varTmp = sqrt(varTmp/num);
	}
	virtual bool trainAssist(){
		if(trainCount == 1){
			meansD = meansTmp;
			varD = varTmp;
		}else{
			meansD += meansTmp;
			varD += varTmp;
		}
		cout<<"\n"<<trainCount;
		return !( trainCount < trainRounds);
	};
	virtual void trainTail(){
		means = meansD/trainRounds;
		var = varD/trainRounds;
		setBestMach();
		cout<<means;
		cout<<var;
	}
public:
	STD(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();
		unsupervise();	
	}
};