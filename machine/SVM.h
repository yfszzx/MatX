template <typename TYPE, bool CUDA>
class SVM:public MachineBase<TYPE, CUDA>{	
private:	
	double * error_cache;
	double * alph;
	TYPE bias;
	TYPE tolScl;//容忍尺度
	TYPE C;
	MatX Win;
	MatX Wout;
	MatXD Woutd;
	MatX hide;
	inline double getError(int k, double E){
		double alp = alph[k];
		double ret;
		if(alp > 0 && alp < C){
			ret = error_cache[k];
		}else{
			ret = learned_func(k) - dt.T[0][k];  //learned_func(int)为非线性的评价函数，即输出函数 
		}
		return ret;
	}
	double kernel_func(int i1,int i2) {  
		/*double s = dot_product_func(i1,i2);   
		s *= -2;   
		s += precomputed_self_dot_product[i1] + precomputed_self_dot_product[i2];   
		return exp(-s / two_sigma_squared);
		*/
	} 
	double learned_func(int k) { 
		int i;   
		double s = 0;   
		for(i = 0; i < dt.trainNum; i++) {  
			if(alph[i] > 0){
				s += alph[i] * dt.T[0][i] * kernel_func(i,k);
			}
		}
		s -= bias; 
		return s;
	}
	bool takeStep(int i1,int i2) {
		if(i1 == i2)  return 0;  //不会优化两个同一样本
		int i;
		float y1,y2,s;
		double alph1,alph2;  //两个乘子的旧值   
		double a1,a2;  //两个乘子的新值    
		double L,H,k11,k22,k12,eta,Lobj,Hobj,delta_b;  
		double gamma;    

		//给变量赋值   
		alph1 = alph[i1];  
		alph2 = alph[i2];  
		y1 = dt.T[0][i1];  
		y2 = dt.T[0][i2];   
		E1 = getError(i1);
		E2 = getError(i2);
		s = y1 * y2;   //计算乘子的上下限 
		if(s == 1)  {  
			gamma = alph1 + alph2;  
			if(gamma > C)   {  
				L = gamma - C; 
				H = C;  
			}   else   {
				L = 0; 
				H = gamma; 
			}  
		}  else  { 
			gamma = alph1 - alph2;  
			if(gamma > 0)   { 
				L = 0;   
				H = C - gamma;   
			}    else   {   
				L = -gamma;   
				H = C;     
			}  
		}   if(L == H) {
			return 0; 
		}
		//计算eta  
		k11 = kernel_func(i1,i1); 
		k22 = kernel_func(i2,i2);
		k12 = kernel_func(i1,i2); 
		eta = 2 * k12 - k11 - k22; 
		if(eta < -0.001)  { 
			double c = y2 * (E2 - E1);   
			a2 = alph2 + c / eta;  //计算新的alph2  
			//调整a2，使其处于可行域  
			if(a2 < L)  a2 = L; 
			else if(a2 > H) {
				a2 = H; 
			}
		}   else {  //分别从端点H,L求目标函数值Lobj,Hobj，然后设a2为所求得最大目标函数值 
			double c1 = eta / 2;
			double c2 = y2 * (E1 - E2) - eta * alph2; 
			Lobj = c1 * L * L + c2 * L;
			Hobj = c1 * H * H + c2 * H;  
			if(Lobj > Hobj + eps)  a2 = L;
			else if(Hobj > Lobj + eps)  a2 = H; 
			else  a2 = alph2;  
		}   
		if(fabs(a2 - alph2) < eps)return 0;  

		a1 = alph1 - s * (a2 - alph2);  //计算新的a1 
		if(a1 < 0)  {  //调整a1,使其符合条件 
			a2 += s * a1; 
			a1 = 0; 
		}   else if(a1 > C)  { 
			a2 += s * (a1 - C);  
			a1 = C; 
		}  
		//更新阀值b   
		double b1,b2,bnew; 
		if (a1 > 0 && a1 < C)  
			bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;  
		else  {  
			if (a2 > 0 && a2 < C)  
				bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;  
			else   {  
				b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
				b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;  
				bnew = (b1 + b2) / 2; 
			} 
		} 
		delta_b = bnew - b; 
		b = bnew;   //对于线性情况，要更新权向量，这里不用了 
		//更新error_cache，对于更新后的a1,a2,对应的位置i1,i2的error_cache[i1] =  error_cache[i2] = 0   
		double t1 = y1 * (a1 - alph1); 
		double t2 = y2 * (a2 - alph2); 
		for(i = 0; i < dt.trainNum; i++)  {
			if(alph[i] > 0 && alph[i] < C)  
				error_cache[i] += t1 * kernel_func(i1,i) + t2 * (kernel_func(i2,i)) - delta_b;
		}
		error_cache[i1] = 0; 
		error_cache[i2] = 0; 
		alph[i1] = a1;   
		alph[i2] = a2;  //存储a1,a2到数组  
		return 1; 
	}   
	//  1：在non-bound乘子中寻找maximum fabs(E1-E2)的样本  
	//	补：在所有不违反KKT条件的乘子中，选择使|E1 −E2|最大的a1进行更新，使得能最大限度增大目标函数的值（类似于梯度下降)
	bool examineFirstChoice(int i1,double E1) {  
		int i2 = -1;  
		double tmax = 0;  
		double E2,temp;   
		for(int k = 0; k < data_num; k++)  {  
			if(alph[k] > 0 && alph[k] < C)   {  
				E2 = error_cache[k];  
				temp = fabs(E1 - E2);   
				if(temp > tmax)    {  
					tmax = temp; 
					i2 = k;  
				}  
			}  
		}   
		if(i2 >= 0 && takeStep(i1,i2)){
			return true; 
		}
		return false; 
	} 
	// 2：如果上面没取得进展,那么从随机位置查找non-boundary样本 
	bool examineNonBound(int i1)  {  
		int k0 = rand() % dt.trainNum; 
		int k,i2; 
		for(k = 0; k < dt.trainNum; k++) { 
			i2 = (k + k0) % dt.trainNum; 
			if((alph[i2] > 0 && alph[i2] < C) && takeStep(i1,i2)){
				return true; 
			}
		}   
		return false; 
	}    
	//  3：如果上面也失败，则从随机位置查找整个样本,(改为bound样本)  
	bool examineBound(int i1) { 
		int k0 = rand() % dt.trainNum; 
		int k,i2;  
		for(k = 0; k < dt.trainNum; k++)    { 
			i2 = (k + k0) % dt.trainNum;  
			if(takeStep(i1,i2)) {
				return true; 
			}
		}    
		return false; 
	} 
	bool examine(int k ){
		double  E = getError(k); 
		double alp = alph[k];  
		double r = dt.T[0][k] * E;   
	//违反KKT条件的判断  
	if((r > tolScl && alp > 0) || (r < -tolScl && alp < C))  { 
		/*    使用三种方法选择第二个乘子   
		1：在non-bound乘子中寻找maximum fabs(E1-E2)的样本   
		2：如果上面没取得进展,那么从随机位置查找non-boundary 样本   
		3：如果上面也失败，则从随机位置查找整个样本,改为bound样本 
		*/  
		if(examineFirstChoice(k, E)){
			return true;  //第1种情况  
		}
		if(examineNonBound(k)){
			return true;  //第2种情况  
		}
		if(examineBound(k)) {
			return true;  //第3种情况 
		}
	}   
	return false; 

	}
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
		Mach<<Win<<Wout;
	};
	virtual void predictCore( MatX * _Y,  MatX* _X, int len = 1){
		MatX ones = MatX::Ones(_X[0].rows());
		_Y[0] = tanh(_X[0] * Win).colJoint(ones) * Wout;
	};
	virtual int getBatchSize(){
		return dt.trainNum  * batchScale;
	}
	virtual void trainHead(){
		error_cache = new double[dt.trainNum];
		alph = new double[dt.trainNum];
	};
	virtual void trainCore(){
		int changedNum = 0; //number of alpha[i], alpha[j] pairs changed in a single step in the outer loop 
		bool allFlag = true;//flag indicating whether the outer loop has to be made on all the alpha[i] 
		while(changedNum > 0 || allFlag)  {  //如果进行全局搜索时依然没有乘子改变，则终止
			changedNum = 0; //记录改变了的乘子个数
			for(k = 0; k < dt.trainNum; k++) {  
				if(allFlag || (alph[k] != 0 && alph[k] != C)){
					if(examine(k)){ //改变lagrange乘子  
						changedNum ++;
					}					
				}
			}  
			if(allFlag){
				allFlag = false; 
			}else  if(changedNum == 0){
				allFlag = true;  //没有发生改变的乘子，则进行全局搜索
			}
		}  
	}
	virtual bool trainAssist(){
		Woutd.add(Wout);	
		dt.Y[0] = hide * Wout;
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
	SVM(dataSetBase<TYPE, CUDA> & dtSet, string path):MachineBase<TYPE, CUDA>(dtSet, path){
		initConfig();		
	};

};
