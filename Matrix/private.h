template <typename TYPE, bool CUDA>
int MatriX< TYPE, CUDA>::elementPos(int _row, int _col) const{
	return rows() * _col + _row; 	
}
template <typename TYPE, bool CUDA>
int MatriX< TYPE, CUDA>::realPos(int idx) const{
	if(!trans()){
		return idx; 
	}else{
		int _row = idx / realCols();
		int _col = idx % realCols();
		return realRows() * _col + _row; 
	}
}
template <typename TYPE, bool CUDA>
bool MatriX<TYPE, CUDA>::sortCmp(int a, int b){
	return a < b;
}
template <typename TYPE, bool CUDA>
void  MatriX<TYPE,CUDA>::removeMapping(vector<int> & map, int * list, const int num, const int removeNum) const{
	sort(list, list + removeNum, sortCmp);
	int listPos = 0;
	for(int i = 0; i< num ; i++){		
		if(listPos == removeNum || i < list[listPos]){			
			map.push_back(i);
		}else{
			do{
				listPos ++;
				if(listPos == removeNum ){
					break;
				}
			}while(i ==  list[listPos]);
		}
	}
}
template <typename TYPE, bool CUDA>
bool  MatriX<TYPE,CUDA>:: matPlusVec(const MatriX<TYPE, CUDA> &mat){
	//处理矩阵和矢量相加的情况
	if(mat.rows()!= rows()){
		if(mat.rows() == 1){
			if(CUDA){
				if(trans()){
					cuWrap::matPlusColVec(dataPrt(), mat.dataPrt(), scale, mat.scale, realRows(), realCols());
				}else{
					cuWrap::matPlusRowVec(dataPrt(), mat.dataPrt(), scale, mat.scale, realRows(), realCols());
				}
			}else{
				if(trans()){
					if(mat.trans()){
						eMat() = eMat() * scale + mat.eMat().replicate(1, rows()) * mat.scale;						
					}else{
						eMat() = eMat() * scale + mat.eMat().transpose().replicate(1, rows()) * mat.scale;
					}
				}else{
					if(mat.trans()){
						eMat() = eMat() * scale + mat.eMat().transpose().replicate(rows(),1) * mat.scale;
					}else{
						eMat() = eMat() * scale + mat.eMat().replicate(rows(),1) * mat.scale;
					}
				}
			}
		}else{
			Assert("矩阵维度不一致,无法进行加法运算");	
		}
		return true;
	}
	if(mat.cols() != cols()){
		if(mat.cols() == 1){
			if(CUDA){
				if(trans()){
					cuWrap::matPlusRowVec(dataPrt(), mat.dataPrt(), scale, mat.scale, realRows(), realCols());
				}else{
					cuWrap::matPlusColVec(dataPrt(), mat.dataPrt(), scale, mat.scale, realRows(), realCols());
				}
			}else{
				if(trans()){
					if(mat.trans()){
						eMat() = eMat() * scale + mat.eMat().replicate(cols(), 1) * mat.scale;
					}else{
						eMat() = eMat() * scale + mat.eMat().transpose().replicate(cols(), 1) * mat.scale;
					}
				}else{
					if(mat.trans()){
						eMat() = eMat() * scale + mat.eMat().transpose().replicate(1, cols()) * mat.scale;
					}else{
						eMat() = eMat() * scale + mat.eMat().replicate(1, cols()) * mat.scale;
					}
				}
			}
		}else{
			Assert("矩阵维度不一致,无法进行加法运算");	
		}
		return true;
	}
	return false;
}