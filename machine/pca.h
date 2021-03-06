template <typename TYPE, bool CUDA>
class PCA:public MachineBase<TYPE, CUDA>{	
private:	
	MatX means;
	MatX dev;
	MatX covMat;
	MatX eigenVals;
	MatX eigenVects;
	

	MatXD meansD;
	MatXD devD;
	MatXD covMatD;
	MatXD meansTmp;
	MatXD devTmp;
	MatXD covMatTmp;
	MatX rotMat;
	int trainRounds;
	float batchNum;
	float varLoss;
	bool showResult;

	bool trainMod;
protected:
	virtual void initConfigValue(){
		initSet(0, "trainRounds",50);
		initSet(1, "batchNum",0.1);	
		initSet(2, "varLoss",0);	
		initSet(3, "showResult",0);	
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
			varLoss = val;
			break;
		case 3:
			showResult = val;
			break;
		}

	};
	virtual void initMachine(){
		Mach<<means<<dev<<covMat<<eigenVals<<eigenVects;
	};
	virtual void predictHead(){
		double varV = 0;
		float varTotal = eigenVals.allSum();
		vector<int >list;
		for(int i = 0; i< dt.inputNum; i++){
			varV += eigenVals[i];
			if(varV >= varTotal * varLoss){
				list.push_back(i);

			}
		}
		rotMat = MatX::Diagonal(dev.cwiseInverse()) * eigenVects.colsMapping(list.data(), list.size());
		unsuperviseDim = list.size();
		cout<<"\n保留方差比例:"<<(1.0f - varLoss)<<"\t保留维度:"<<unsuperviseDim<<"/"<<inputNum;
	};
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){	
		for(int i = 0; i< len; i++){
			_Y[i] = (_X[i] - means) * MatX::Diagonal(dev.cwiseInverse());
			_Y[i] = _Y[i] * eigenVects;
		}
		
	};
	virtual int getBatchSize(){
		return batchNum;
	}

	virtual void trainCore(){
		int num = (dt.seriesLen - dt.preLen) * dt.X[0].rows();
		meansTmp = dt.X[dt.preLen].sum();
		for(int i = dt.preLen + 1; i<dt.seriesLen; i++){
			meansTmp.add(dt.X[i].sum());
		}
		meansTmp /= num;
		means = meansTmp;


		MatX * dtCpy = new MatX[dt.seriesLen - dt.preLen];
		dtCpy[0] = dt.X[dt.preLen] - means;
		devTmp = square(dtCpy[0]).sum();
		for(int i =1; i< dt.seriesLen - dt.preLen; i++){
			dtCpy[i] = dt.X[i + dt.preLen] - means;
			devTmp.add(square(dtCpy[i]).sum());
		}
		devTmp = sqrt(devTmp/num);

		covMatTmp = dtCpy[0].T() * dtCpy[0];
		for(int i = 1; i< dt.seriesLen - dt.preLen; i++){
			covMatTmp.add(dtCpy[i].T() * dtCpy[i]);
		}

		covMatTmp /= num;		
		MatXD tmp = (devTmp.T() * devTmp).cwiseInverse();
		covMatTmp = covMatTmp.cwiseProduct(tmp);//归一化方差
		delete [] dtCpy;
	}
	virtual bool trainAssist(){
		if(trainCount == 1){
			meansD = meansTmp;
			devD = devTmp;
			covMatD = covMatTmp;
		}else{
			meansD += meansTmp;
			devD += devTmp;
			covMatD += covMatTmp;
		}
		cout<<"\n"<<trainCount;
		return !( trainCount < trainRounds);
	};
	virtual void trainTail(){
		means = meansD/trainRounds;
		dev = devD/trainRounds;
		covMat = covMatD/trainRounds;
		eigenVects = covMat.eigenSolver(eigenVals);		
		if(showResult){
			showParams();
		}
		setBestMach();
	}
public:
	void showParams(){
		cout<<"\nmean:";
		cout<<means;
		getchar();
		cout<<"\ndev:";
		cout<<dev;
		getchar();
		cout<<"\neigenValues:";
		cout<<eigenVals.T();
		getchar();
		cout<<"\ncovarianceMatrix:";
		cout<<covMat;
		getchar();
		cout<<"\nvectorMatrix:";
		cout<<eigenVects;
		getchar();
	}
	PCA(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();
		unsupervise();	
	}
	MatX tmp;
	MatX tmp2;
	MatX tmp3;
};
