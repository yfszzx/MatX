template <typename TYPE, bool CUDA>
class ESN:public MachineBase<TYPE, CUDA>{	
private:	
	MatX Win;
	MatX Wjnt;
	MatX Wout;
	
	int nodes;	
	float regOut; //岭回归系数
	float spectralRadius; //连接矩阵谱半径
	float sparseDegree; //连接矩阵稀疏度

	int hNum;
protected:
	virtual void initConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut", 0);
		initSet(2, "SR", 0.9);
		initSet(3,  "SD", 0.1);
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
			spectralRadius = val;
			break;
		case 3:
			sparseDegree = val;
			break;
		}
	};
	virtual void initWs(bool trainMod = true){
		hNum = inputNum + nodes + 1;
		Win = MatX::Random(inputNum, nodes);
		Wjnt = MatX::Random(nodes, nodes);
		Wout  = MatX::Random(hNum, outputNum);
		Ws<<Win<<Wjnt<<Wout;		
	};
	virtual void predict( MatX * _Y,  MatX * _X, int len){
		MatX ones = MatX::Ones(_X[0].rows());
		MatX H =  tanh(_X[0] * Win );
		MatX nilH = MatX::Zero(H.rows(), H.cols());
		_Y[0] = nilH.colJoint(_X[0]).colJoint(ones) * Wout;
		for(int i = 1; i< len; i++){			
			_Y[i] = H.colJoint(_X[i]).colJoint(ones) * Wout;
			H = tanh(_X[i] * Win + H * Wjnt);
		}
	};
public:
	ESN(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();
	};
	virtual void train(){
		dt.makeBatch(dt.trainNum);
		dt.loadBatch();

		//生成系数的、特定谱半径的连接矩阵
		MatX tmp = (MatX::Random(nodes, nodes) > MatX::Constant(nodes, nodes,  (1 - sparseDegree * 2)));
		Wjnt = Wjnt.cwiseProduct(tmp);
		Wjnt = Wjnt / Wjnt.spectralRadius() * spectralRadius;
	
		MatX * HX = new MatX[dt.seriesLen];	
		MatX ones = MatX::Ones(dt.X[0].rows());
		MatX I = MatX::eye(hNum);	
		MatX A = MatX::Zero(hNum, hNum);		
		MatX H = tanh(dt.X[0] * Win );
		for(int i = 1; i < dt.seriesLen; i++){
			HX[i] = H.colJoint(dt.X[i]).colJoint(ones);
			H = tanh(dt.X[i] * Win + H * Wjnt);
			if(i >= dt.preLen){
				A += I * regOut + HX[i].T() * HX[i];
			}
		}
		Wout = 0;
		A = A.inv();
		TYPE loss = 0;
		for(int i = dt.preLen; i < dt.seriesLen; i++){
			Wout += A * HX[i].T() * dt.T[i];
			dt.Y[i] = HX[i] * Wout;
			loss += (dt.Y[i] - dt.T[i]).norm2();
		}
		loss /= (dt.seriesLen - dt.preLen);

		bestWs = Ws;
		cout<<"\ndataLoss"<<loss/dt.batchSize/2/dt.batchInitLoss;
		cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;
		dt.showResult();
	};
};
