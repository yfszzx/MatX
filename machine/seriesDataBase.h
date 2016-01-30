template<typename TYPE, bool CUDA>
class seriesDataBase:public dataSetBase<TYPE, CUDA>{
protected:
	TYPE * srcX;
	TYPE * srcT;
	virtual int loadSeries() = 0;//·µ»ØdataNum;
	virtual void cutSeries(int dtNum){	
		TYPE *tmpX = new TYPE[dtNum * inputNum * seriesLen];
		TYPE *tmpT = new TYPE[dtNum * outputNum * seriesLen];
		for(int i = 0; i < seriesLen; i++){
			for(int j = 0; j < dtNum; j++){
				int srcPos = j * (seriesLen - preLen) + i;
				int dstPos = i * dtNum + j;
				memcpy(tmpX + dstPos * inputNum,  srcX + srcPos * inputNum, sizeof(TYPE) * inputNum);
				memcpy(tmpT + dstPos * outputNum,  srcT + srcPos * outputNum, sizeof(TYPE) * outputNum);				
			}
		}
		swap(srcX, tmpX);
		swap(srcT, tmpT);
		delete [] tmpX;
		delete [] tmpT;
	}
	void loadData(){	
		for(int i = 0; i < seriesLen; i++){
			X0[i].importData(srcX + dataNum * inputNum * i);
			T0[i].importData(srcT + dataNum * outputNum * i);	
		}
		
	}
	void load(){
		int num = loadSeries();
		dataSetBase<TYPE, CUDA>::load(num);		
	}
	
public:
	seriesDataBase(int _preLen){
		srcX = NULL;
		srcT = NULL;
		preLen = _preLen;
	};
	~seriesDataBase(){
		if(srcX != NULL){
			delete [] srcX;
			delete [] srcT;
		}
	}
	void show(){
		dataSetBase::show();
		cout<<"\nseriesLen:"<<seriesLen<<"\tpreLen:"<<preLen;
	}
};