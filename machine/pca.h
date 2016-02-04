template <typename TYPE, bool CUDA>
class PCA:public MachineBase<TYPE, CUDA>{	
private:	
	MatX means;
	MatX var;
	MatX covMat;
	MatX eigenVals;
	MatX eigenVects;
	MatX rotMat;

	MatXD meansD;
	MatXD varD;
	MatXD covMatD;
	MatXD meansTmp;
	MatXD varTmp;
	MatXD covMatTmp;
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
			Mach<<means<<var<<covMat<<eigenVals<<eigenVects<<rotMat;
	};
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){
		//MatX ones = MatX::Ones(_X[0].rows());
		//_Y[0] = (_X[0] - means) * transMat;
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

		MatX * dtCpy = new MatX[dt.seriesLen - dt.preLen];
		dtCpy[0] = dt.X[dt.preLen] - means;
		varTmp = square(dtCpy[0]).sum();
		for(int i =1; i< dt.seriesLen - dt.preLen; i++){
			dtCpy[i] = dt.X[i + dt.preLen] - means;
			varTmp.add(square(dtCpy[i]).sum());

		}
		varTmp = sqrt(varTmp/num);
		covMatTmp = dtCpy[0].T() * dtCpy[0];
		for(int i = 1; i< dt.seriesLen - dt.preLen; i++){
			covMatTmp.add(dtCpy[i].T() * dtCpy[i]);
		}
		covMatTmp /= num;		
		MatXD tmp = (varTmp.T() * varTmp).cwiseInverse();
		covMatTmp = covMatTmp.cwiseProduct(tmp);//¹éÒ»»¯·½²î
		delete [] dtCpy;
	}
	virtual bool trainAssist(){
		if(trainCount == 1){
			meansD = meansTmp;
			varD = varTmp;
			covMatD = covMatTmp;
		}else{
			meansD += meansTmp;
			varD += varTmp;
			covMatD += covMatTmp;
		}
		cout<<"\n"<<trainCount;
		return !( trainCount < trainRounds);
	};
	virtual void trainTail(){
		means = meansD/trainRounds;
		var = varD/trainRounds;
		covMat = covMatD/trainRounds;
		cout<<means;
		cout<<var;
		cout<<covMat;
	
		eigenVects = covMat.eigenSolver(eigenVals);
		cout<<"\neigenVals:";
		cout<<eigenVals;
		cout<<"\neigenVects";
		cout<<eigenVects;
		rotMat = eigenVects.T().cwiseQuotient(var.replicate(var.cols(), 1) );
		MatX T = (dt.X[0] - means) * rotMat;
		MatX TT = T * rotMat.T();
		cout<<T -TT;
		getchar();
		setBestMach();
	}
public:
	PCA(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();		
	};
	
};
