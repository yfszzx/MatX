template <typename TYPE, bool CUDA>
class ELM:public MachineBase<TYPE, CUDA>{	
private:	
	MatX Win;
	MatX Wout;
	MatXD Woutd;
	TYPE regOut;
	int nodes;
	int trainRounds;
	float batchScale;
protected:
	virtual void initConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut",0);
		initSet(2, "trainRounds",20);
		initSet(3, "batchScale",0.5);
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
	virtual void initMachine(){
		Win = MatX::Random(inputNum, nodes);
		Wout =  MatX::Zero(nodes + 1, outputNum);
		Woutd = Wout;
		Mach<<Win<<Wout;
	};
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){
		MatX ones = MatX::Ones(_X[0].rows());
		_Y[0] = tanh(_X[0] * Win).colJoint(ones) * Wout;
	};
	virtual int getBatchSize(){
		return dt.trainNum  * batchScale;
	}
	virtual void trainHead(){};
	virtual void trainCore(){
		MatX I = MatX::Identity(nodes + 1);	
		MatX A;
		MatX ones = MatX::Ones(dt.X[0].rows());
		MatX hide =  tanh(dt.X[0]* Win).colJoint(ones);
		A = I * regOut + hide.T() * hide;
		Wout = A.inv() * hide.T() * dt.T[0];
		dt.Y[0] = hide * Wout;

	}
	virtual bool trainAssist(){
		Woutd.add(Wout);		
		cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/batchSize/2/batchInitLoss;		
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();
		return !( trainCount < trainRounds);
	};
	virtual void trainTail(){
		Wout = Woutd/trainRounds;
		setBestMach();
	}
public:
	ELM(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();		
	};
	
};
