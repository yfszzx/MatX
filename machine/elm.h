template <typename TYPE, bool CUDA>
class ELM:public MachineBase<TYPE, CUDA>{	
private:	
	MatX Win;
	MatX Wout;
	MatX hide;
	TYPE regOut;
	int nodes;
	int trainRounds;
	float batchSize;
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
			batchSize = val;
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
public:
	ELM(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();
	};
	virtual void train(){
		int rnd = dt.randBatch?trainRounds:1;
		MatX I = MatX::Identity(nodes + 1);	
		MatX A;
		MatX ones;
		dt.makeBatch(batchSize * dt.trainNum);
		for(int i = 0; i< rnd; i++){
			dt.loadBatch();
			pthread_t tid1, tid2;
			void *ret1,*ret2;
			ones= MatX::Ones(dt.X[0].rows());
			hide =  tanh(dt.X[0]* Win).colJoint(ones);
			A = I * regOut + hide.T() * hide;
			MatX W = A.inv() * hide.T() * dt.T[0];
			Wout += W;
			dt.Y[0] = hide * W;
			cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/dt.batchSize/2/dt.batchInitLoss;
			cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
			dt.showResult();
			dt.makeBatch(batchSize * dt.trainNum);
			
		}
		Wout /= rnd;
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();

	};
};
