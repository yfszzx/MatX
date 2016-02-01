template <typename TYPE, bool CUDA>
class ELM:public MachineBase<TYPE, CUDA>{	
private:	
	MatX Win;
	MatX Wout;
	TYPE regOut;
	int nodes;
	int trainRounds;
	float batchScale;
protected:
	virtual void initConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut",0);
		initSet(2, "trainRounds",20);
		initSet(3, "batchSize",0.5);
	};
	virtual void setConfigValue(int idx, float val){
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
	};
	virtual void initWs(bool trainMod = true){
		Win = MatX::Random(inputNum, nodes);
		Wout =  MatX::Zero(nodes + 1, outputNum);
		Ws<<Win<<Wout;
	};
	virtual void predict( MatX * _Y,  MatX* _X, int len = 1){
		MatX ones = MatX::Ones(_X[0].rows());
		_Y[0] = tanh(_X[0] * Win).colJoint(ones) * Wout;
	};
	virtual int getBatchSize(){
		return dt.trainNum  * batchScale;
	}
public:
	ELM(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();		
	};
	virtual void train(){
		MatX I = MatX::Identity(nodes + 1);	
		MatX A;
		MatX ones = MatX::Ones(dt.X[0].rows());
		MatX hide =  tanh(dt.X[0]* Win).colJoint(ones);
		A = I * regOut + hide.T() * hide;
		Wout = A.inv() * hide.T() * dt.T[0];
		dt.Y[0] = hide * Wout;
	
	}
	virtual bool trainOperate(){
		cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/batchSize/2/batchInitLoss;
		averageWs(bestWs, Ws, trainCount);
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();
		return !( trainCount < trainRounds);
	};
};
