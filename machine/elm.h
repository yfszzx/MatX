template <typename TYPE, bool CUDA>
class ELM:public MachineBase<TYPE, CUDA>{	
private:	
	MatX Win;
	MatX Wout;
	MatXD Woutd;
	MatX hide;

	MatX C;	
	TYPE regOut;
	int nodes;
	int trainRounds;
	float batchNum;
	int showFreq;
protected:
	virtual void initConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut",0);
		initSet(2, "trainRounds",20);
		initSet(3, "batchNum",10000);
		initSet(4, "showFreq",10);
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
			batchNum = val;
			break;
		case 4:
			showFreq = val;
			break;
		}
	};
	virtual void initMachine(){
		Win = MatX::Random(inputNum, nodes);
		Wout =  MatX::Zero(nodes, outputNum);
		
		Mach<<Win<<Wout;
	};
	virtual int getBatchSize(){
		return batchNum;
	}
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){
		_Y[0] = (_X[0] * Win).tanh() * Wout;
	};
	virtual void trainHead(){
		Woutd = MatX::Zero(nodes, outputNum);
		C = MatX::eye(nodes) * regOut;	
	};
	virtual void trainCore(){
		hide = (dt.X[0] * Win).tanh();
		MatX A = C + hide.T() * hide;
		Wout = A.inv() * hide.T() * dt.T[0];
	}
	virtual bool trainAssist(){
		Woutd.add(Wout);	
		dt.Y[0] = hide * Wout;		
		cout<<"\n"<<trainCount;
		cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/batchSize/2/batchInitLoss;
		if(trainCount % showFreq == showFreq - 1){
			Wout = Woutd/trainCount;
			cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;		
		}		
		return !( trainCount < trainRounds);
	};
	virtual void trainTail(){
		Wout = Woutd/trainRounds;
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;		
		setBestMach();
		dt.showResult();
	}
public:
	ELM(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();		
	};
	
};
