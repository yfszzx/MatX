template <typename TYPE, bool CUDA>
class MLP:public ANNBase<TYPE, CUDA>{
protected:
	MatriX<TYPE, CUDA> Win;
	MatriX<TYPE, CUDA> Wout;
	MatriX<TYPE, CUDA> Bin;
	MatriX<TYPE, CUDA> Bout;
	MatriX<TYPE, CUDA> grad_out;
	MatriX<TYPE, CUDA> grad_in;
	MatriX<TYPE, CUDA> grad_Bout;
	MatriX<TYPE, CUDA> grad_Bin;	
	TYPE regIn;
	TYPE regOut;
	virtual void forward();
	virtual void backward();
	virtual void initWs(bool trainMod = true);
	virtual void predict( MatriX<TYPE, CUDA> * _Y,  MatriX<TYPE, CUDA>* _X, int len = 1) ;
public:
	
	MLP(int _nodes, dataSetBase<TYPE, CUDA> & dtSet);
	virtual void setRegulars(TYPE * regVal);
};