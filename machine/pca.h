template <typename TYPE, bool CUDA>
class ELM:public MachineBase<TYPE, CUDA>{	
private:	
	MatX means;
	MatX transMat;
	TYPE regOut;
	int nodes;
	int trainRounds;
	float batchScale;
protected:
	virtual void initConfigValue(){
		/*
		initSet(0, "nodes",100);
		initSet(1, "regOut",0);
		initSet(2, "trainRounds",20);
		*/
		initSet(3, "batchSize",0.5);
		
	};
	virtual void setConfigValue(int idx, float val){
		/*
		switch(idx){
		case 0:
			nodes = val;
			break;
		case 1:
			regOut = val;
			break;
		case 2:
			trainRounds = val;
			break;
		case 3:
			batchScale = val;
			break;
		}
		*/
	};
	virtual void initWs(bool trainMod = true){
		//means = MatX::Random(inputNum, nodes);
		//Wout =  MatX::Zero(nodes + 1, outputNum);
		Ws<<means<<transMat;
	};
	virtual void predict( MatX * _Y,  MatX* _X, int len = 1){
		MatX ones = MatX::Ones(_X[0].rows());
		_Y[0] = (_X - means) * transMat;
	};
	virtual int getBatchSize(){
		return dt.trainNum  * batchScale;
	}
public:
	ELM(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();		
	};
	virtual void train(){
		if(trainCount < trainRounds){
			means += dt.X[0].sum();

		}else if(trainCount < 2 *trainRounds){
			var += (dt.X[0] - means).norm2();
		}else if(trainCount < 3 * trainRounds){
			cov += (dt.X[0] - means) *  (dt.X[0] - means).T();
		}
	}
	virtual bool trainOperate(){
		cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/batchSize/2/batchInitLoss;
		averageWs(bestWs, Ws, trainCount);
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();
		return !( trainCount < trainRounds);
	};
};