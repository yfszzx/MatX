template<typename TYPE, bool CUDA>
class seriesDataBase:public dataSetBase<TYPE, CUDA>{
private:
	int smartSeriesLen(int seriNum, int len){
		int num = (len - preLen) / (seriesLen - preLen);
		int left = (len - preLen) % (seriesLen - preLen);
		seriesLen  += left%num;
		num = (len - preLen)/(seriesLen - preLen);
		return num * seriNum;
	}
protected:
	void loadDataSet(const TYPE * _X, const TYPE * _T, int seriNum, int len){
		dataNum = smartSeriesLen(seriNum, len);
		TYPE *tmpX = new TYPE[dataNum * inputNum * seriesLen];
		TYPE *tmpT = new TYPE[dataNum * outputNum * seriesLen];
		int subNum = dataNum / seriNum;
		int realLen = seriesLen - preLen;
		for(int i = 0; i < seriesLen; i++){
			for(int j = 0; j < seriNum; j++){
				for(int k = 0; k < subNum; k++){
					int dataIdx = k * seriNum + j;
					int srcPos = j * len + k * realLen;
					int dstPos = i * dataNum + dataIdx;
					memcpy(tmpX + dstPos * inputNum,  _X + srcPos * inputNum, sizeof(TYPE) * inputNum);
					memcpy(tmpT + dstPos * outputNum,  _T + srcPos * outputNum, sizeof(TYPE) * outputNum);	

				}
						
			}
		}
		initDataSpace();
		for(int i = 0; i < seriesLen; i++){
			X0[i].importData(tmpX + dataNum * inputNum * i);
			T0[i].importData(tmpT + dataNum * outputNum * i);	
		}	
		delete [] tmpX;
		delete [] tmpT;
	}
public:
	seriesDataBase(){
		preLen = 1;
		seriesMod = true;		
	};
	void show(){
		dataSetBase::show();
		cout<<"\nseriesLen:"<<seriesLen<<"\tpreLen:"<<preLen;
		
	}
};