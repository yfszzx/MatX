template <typename TYPE, bool CUDA>
class ELM:public MachineBase<TYPE, CUDA>{	
private:	
	MatX Win;
	MatX Wout;
	MatX hide;
	TYPE regOut;
	int nodes;
protected:
	virtual void initWs(bool trainMod = true){
		Win = MatX::Random(inputNum, nodes);
		Ws<<Win<<Wout;		
	};
	virtual void predict( MatX * _Y,  MatX* _X, int len){
		MatX ones = MatX::Ones(_X[0].rows());
		_Y[0] = tanh(_X[0] * Win).colJoint(ones) * Wout;
	};
public:
	ELM(int _nodes, dataSetBase<TYPE, CUDA> & dtSet):MachineBase<TYPE, CUDA>(dtSet){
		nodes = _nodes;
		regOut = 0;
	};
	~ELM(){};
	virtual void train(){
		dt.makeBatch(dt.trainNum);
		dt.loadBatch();

		MatX ones = MatX::Ones(dt.X[0].rows());
		hide =  tanh(dt.X[0]* Win).colJoint(ones);
		MatX I = MatX::Identity(nodes + 1);	
		MatX A = I * regOut + hide.T() * hide;
		Wout = A.inv() * hide.T() * dt.T[0];
		dt.Y[0] = hide * Wout;
		bestWs = Ws;

		cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/dt.batchSize/2/dt.batchInitLoss;
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();
	};
	virtual void setRegulars(TYPE * regVal){
		regOut = regVal[0];
	};	
	virtual void trainSet(int con, float val){
	};	

};
