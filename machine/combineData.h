template <typename TYPE, bool CUDA>
class combineData: public dataSetBase<TYPE, CUDA>{
public:
	int dtNum;
	combineData(int num){
		init(1, 1);
		set("activeFunc", LINEAR);
		dtNum = num;
	}
	void createSamples(){		
		TYPE *tX = new TYPE[dtNum];
		TYPE *tT = new TYPE[dtNum];
		for(int i = 0; i<dtNum; i++){
			tX[i] = 3.1415 * TYPE(Mrand(10000) -5000)/10000;
			tT[i] = sin(tX[i]);
		}
		loadSamples(tX, tT, dtNum);
		delete [] tX;
		delete [] tT;
	}
	void showResult(){
		cout<<"\ncorrel:"<<Yv[0].correl(Tv[0]);
	}
	virtual void showValidsResult(MatGroup<TYPE, CUDA> &T, MatGroup<TYPE, CUDA> &Y){
		cout<<"\ncorr:"<<T.correl(Y);
	};
};