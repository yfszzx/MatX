template <typename TYPE, bool CUDA>
class RNN:public ANNBase<TYPE, CUDA>{
protected:
	MatriX<TYPE, CUDA> Win;
	MatriX<TYPE, CUDA> Wout;
	MatriX<TYPE, CUDA> Bin;
	MatriX<TYPE, CUDA> Bout;
	MatriX<TYPE, CUDA> grad_out;
	MatriX<TYPE, CUDA> grad_in;
	MatriX<TYPE, CUDA> grad_Bout;
	MatriX<TYPE, CUDA> grad_Bin;	
	MatriX<TYPE, CUDA> Wjnt;
	MatriX<TYPE, CUDA> grad_jnt;
	TYPE regIn;
	TYPE regOut;
	TYPE regJnt;
	virtual void forward();
	virtual void backward();
	virtual void initWs(bool trainMod = true);
	virtual void predict( MatriX<TYPE, CUDA> * _Y,  MatriX<TYPE, CUDA>* _X, int len = 1) ;

public:	
	RNN(int _nodes, seriesDataBase<TYPE, CUDA> & dtSet);	
	virtual void setRegulars(TYPE * regVal);
};