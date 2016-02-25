#include <fastMLP/fastMLP.h> 
template <typename TYPE, bool CUDA>
class fastMLP:public MachineBase<TYPE, CUDA>, search_tool{	
private:	
	MatX W;
	multi_perceptrons *mlp;
	int nodes;
	int trainRounds;
	float batchScale;
	float regOut;
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
		string path = "ss";
		mlp=new multi_perceptrons(path);
		mlp->train_mod=true;
		mlp->struct_simple_set(dt.inputNum, dt.outputNum, nodes, 't', 's');
	};
	virtual int getBatchSize(){
		return dt.trainNum  * batchScale;
	}
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){
		
	};
	virtual void trainHead(){
			mlp->train_mod=true;
			search_init(mlp->weight_len, &(mlp->result), mlp->weight, mlp->deriv);
			set_search();
		
	};
	virtual void trainCore(){
		mlp->set_data_num(batchSize);		
		search(1);
		dt.Y[0] = MatX::Zero(dt.outputNum, batchSize);
		dt.Y[0].importData(mlp->nerv_top->output);
		dt.Y[0] = dt.Y[0].T();
	}
	virtual bool trainAssist(){
		//cout<<"\ndataLoss"<<(dt.Y[0] -dt.T[0]).squaredNorm()/batchSize/2/batchInitLoss;		
		//cout<<"\nLoss:"<<getValidLoss()/dt.validInitLoss;		
		return !( trainCount < trainRounds);
	};
	virtual void trainTail(){
		//setBestMach();
		//dt.showResult();
	}
	virtual bool show_and_control(int i){
		//Dbg2(mlp->result, dt.batchInitLoss);
		cout<<"\n"<<i<<" "<<mlp->result;///dt.batchInitLoss/2;
		return true;
	}
	virtual void cacul(){
		TYPE * tX = dt.X[0].T().getData(true);
		TYPE * tT = dt.T[0].T().getData(true);
		mlp->cacul_nerv(tX, tT);		
		cuWrap::free(tX);
		cuWrap::free(tT);
		//mlp->cacul_nerv(dt.X[0].T().memPrt(), dt.T[0].T().memPrt());
	}
public:
	fastMLP(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path),search_tool(path){
		initConfig();		
	};

};
template <typename TYPE, bool CUDA>
class fMLP:public ANNBase<TYPE, CUDA>{
protected:
	string fold;
	MatX W;
	MatX gd;
	int nodes;
	TYPE regIn;
	TYPE regOut;
	multi_perceptrons *mlp;
	virtual void annInitConfigValue(){
		initSet(0, "nodes",100);
		initSet(1, "regOut", 0);
		initSet(2, "regIn", 0);
	}
	virtual void annSetConfigValue(int idx, float val){
		switch(idx){
		case 0:
			nodes = val;
			break;
		case 1:
			regOut = val;
			break;
		case 2:
			regIn = val;
			break;
		}
	}
	virtual void initMachine(){		
		mlp = new multi_perceptrons("flod");
		mlp->train_mod = true;
		mlp->struct_simple_set(dt.inputNum, dt.outputNum, nodes, 't', 's');
		W = MatX::Random(mlp->weight_len,1)/1000;
		W.exportData(mlp->weight, true);
		Mach<<W;
	}
	virtual void predictCore( MatX * _Y, MatX * _X, int len = 1) {
		_Y[0] = MatX::Zero(dt.outputNum, _X[0].rows());
		TYPE * tX = _X[0].T().getData(true);
		_Y[0].importData(mlp->layer_out(mlp->layers_num - 1, tX, _X[0].rows()), true);
		_Y[0] = _Y[0].T();
		cuWrap::free(tX);	
	}
	virtual void annTrainHead(){
		grads.clear();
		gd = MatX::Zero(mlp->weight_len);
		grads<<gd;
	}
	virtual void forward(){
		//gd.exportData(mlp->deriv, true);
		W.exportData(mlp->weight, true);
		mlp->set_data_num(batchSize);		
		TYPE * tX = dt.X[0].T().getData(true);
		TYPE * tT = dt.T[0].T().getData(true);
		MatX tt = MatX::Zero(dt.inputNum, 5);
		Dbg(dt.X[0].topRows(5).T());
		tt.importData(tX, true);
		float * j = new float[784];
		cuWrap::memD2H(j, tX, sizeof(float) *784);
		for(int i = 0; i < 784; i++){
			cout<<j[i]<<"\t";
		}
		getchar();
		Dbg(tt);
		Dbg(dt.T[0].topRows(5).T());
		MatX ttt = MatX::Zero(dt.outputNum, 5);
		ttt.importData(tT, true);
		Dbg(ttt);
		mlp->cacul_nerv(tX, tT);		
		cuWrap::free(tX);
		cuWrap::free(tT);
	}
	virtual void backward(){
		dt.Y[0] = MatX::Zero(dt.outputNum, batchSize);
		dt.Y[0].importData(mlp->nerv_top->output, true);
		dt.Y[0] = dt.Y[0].T();
		gd.importData(mlp->deriv, true);
		W.importData(mlp->weight, true);
		loss = mlp->result;
		//gd /=2;
		//cuWrap::scale(mlp->deriv, 0.5f, mlp->weight_len);
		//dataLoss = mlp->real_result/2;
	}
public:	
	fMLP(dataSetBase<TYPE, CUDA> & dtSet, string path):ANNBase(dtSet, path){
		initConfig();
		fold = path;
	};
};