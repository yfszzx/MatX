
template <typename TYPE, bool CUDA>
class noise:public MachineBase<TYPE, CUDA>{	
private:	
	MatX randMat;
	MatX nullMat;
	float stdDev;
protected:
	virtual void initConfigValue(){
		initSet(0, "stdDev", 0.2);
	};
	virtual void setConfigValue(int idx, float val){
		switch(idx){
		case 0:
			stdDev = val;
			break;
		}

	};
	virtual void initMachine(){
		nullMat = MatX::Zero(1);
		Mach<<nullMat;
	};
	virtual void predict( MatX * _Y,  MatX* _X, int len = 1){	
		if (!use) {
			for (int i = 0; i < len ; ++i) {
				_Y[i] = _X[i];	
			}
			return;
		}
		int col = _X[0].cols();
		int row = _X[0].rows();
		for(int i = 0; i< len; i++){
			_Y[i] = _X[i] + MatX::NormalRandom(row, col, stdDev, 0);
		}

	};
	virtual int getBatchSize(){ return 1; }
	virtual void trainCore(){ }
	virtual bool trainAssist(){ return 1; };
	virtual float trainTail(){ return 1; }
public:
	bool use;
	void showParams(){ }
	noise(dataSetBase<TYPE, CUDA> & dtSet, string path, int validId):MachineBase<TYPE, CUDA>(dtSet, path, validId){
		initConfig();
		unsupervise();	
		use = true;
	}

};


