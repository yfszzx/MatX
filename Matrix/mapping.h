template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::replicate(int verticalNum, int horizontalNum) const{
	int r = rows() * verticalNum;
	int c = cols() * horizontalNum;
	if(trans()){
		swap(r,c);
		swap(verticalNum,horizontalNum);
	}
	MatriX<TYPE, CUDA> ret(r, c);
	if(CUDA){
		cuWrap::replicate(ret.dataPrt(), dataPrt(), realRows(), realCols(), verticalNum, horizontalNum);
	}else{
		ret.eMat() = eMat().replicate(verticalNum, horizontalNum);
	}
	ret.scale = scale;
	ret.setTrans(trans());
	return ret;
};

template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::colsMapping(int *list, int num) const{
	int retRows;
	int retCols;
	if(!trans()){
		retRows = rows();
		retCols = num;
	}else{
		retRows = num;
		retCols = rows();
	}
	MatriX<TYPE, CUDA> ret(retRows, retCols);
	if(CUDA){
		if(!trans()){
			cuWrap::colsMapping(ret.dataPrt(), dataPrt(), list, num, realRows(), realCols());
		}else{
			cuWrap::rowsMapping(ret.dataPrt(), dataPrt(), list, num, realRows(), realCols());
		}

	}else{
		if(!trans()){
			for(int i = 0; i < num; i++){
				memcpy(ret.dataPrt() + retRows * i, dataPrt() + retRows * list[i], sizeof(TYPE) * retRows);		
			}
		}else{
			TYPE *  retP = ret.dataPrt();
			TYPE *  P = dataPrt();
			for(int i = 0; i < num; i++){
				for(int j = 0; j < retCols; j++){
					retP[retRows * j + i] = P[realRows() * j + list[i]];
				}
			}
		}

	}
	ret.scale = scale;
	ret.setTrans(trans());
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::rowsMapping(int *list, int num) const{
	int retRows;
	int retCols;
	if(!trans()){
		retRows = num;
		retCols = cols();
	}else{
		retRows = cols();
		retCols = num;
	}
	MatriX<TYPE, CUDA> ret(retRows, retCols);
	if(CUDA){
		if(!trans()){
			cuWrap::rowsMapping(ret.dataPrt(), dataPrt(), list, num, realRows(), realCols());
		}else{
			cuWrap::colsMapping(ret.dataPrt(), dataPrt(), list, num, realRows(), realCols());
		}

	}else{
		if(!trans()){
			TYPE *  retP = ret.dataPrt();
			TYPE *  P = dataPrt();
			for(int i = 0; i < num; i++){
				for(int j = 0; j < retCols; j++){
					retP[retRows * j + i] = P[realRows() * j + list[i]];
				}
			}			
		}else{
			for(int i = 0; i < num; i++){
				memcpy(ret.dataPrt() + retRows * i, dataPrt() + retRows * list[i], sizeof(TYPE) * retRows);
			}
		}

	}
	ret.scale = scale;
	ret.setTrans(trans());
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::col(int _col) const{
	return colsMapping(&_col, 1);	
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::row(int _row) const{
	return rowsMapping(&_row, 1);
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::topRows(int num) const{
	return removeBottomRows(rows() - num);
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::bottomRows(int num) const{
	return removeTopRows(rows() - num);
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::leftCols(int num) const{
	return removeRightCols(cols() - num);

};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::rightCols(int num) const{
	return removeLeftCols(cols() - num);
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeTopRows(int num) const{
	MatriX<TYPE, CUDA> ret(cols(), rows() - num);
	MatriX<TYPE, CUDA> tmp(this->T(), TRAN);
	ret.loadMat(tmp.dataPrt() + num * tmp.rows(), CUDA);
	ret.scale = scale;
	return ret.transpose();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeBottomRows(int num) const{
	MatriX<TYPE, CUDA> ret( cols(), rows() - num);
	MatriX<TYPE, CUDA> tmp(this->T(), TRAN);
	ret.loadMat(tmp.dataPrt() , CUDA);
	ret.scale = scale;
	return ret.transpose();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeLeftCols(int num) const{
	MatriX<TYPE, CUDA> ret(rows(), cols() - num);
	MatriX<TYPE, CUDA> tmp(*this, TRAN);
	ret.loadMat(tmp.dataPrt() + num * tmp.rows(), CUDA);
	ret.scale = scale;
	return ret;

};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeRightCols(int num) const{
	MatriX<TYPE, CUDA> ret(rows(), cols() - num);
	MatriX<TYPE, CUDA> tmp(*this, TRAN);
	ret.loadMat(tmp.dataPrt(), CUDA);
	ret.scale = scale;
	return ret;
};

template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeCols(int * list, int num) const{		
	vector<int> map;
	removeMapping(map, list, cols(), num);	
	return colsMapping(map.data(), map.size());
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeRows(int * list, int num) const{		
	vector<int> map;
	removeMapping(map, list, rows(), num);
	return rowsMapping(map.data(), map.size());
}
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeCol(int idx) const{
	return removeCols(&idx, 1);
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::removeRow(int idx) const{
	return removeRows(&idx, 1);
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::rowJoint(const MatriX<TYPE, CUDA> & mat)  const{
	if(mat.cols()!= cols()){
		Assert("矩阵行数不一致，无法连接");
	}
	int rc = cols();
	int rr = rows() + mat.rows();
	MatriX<TYPE, CUDA> ret(rc, rr);
	ret.setTrans();
	MatriX<TYPE, CUDA> tmp(this->T(), ALL);
	if(CUDA){
		cuWrap::memD2D(ret.dataPrt(), tmp.dataPrt(), sizeof(TYPE) * size());
	}else{
		memcpy(ret.dataPrt(), tmp.dataPrt(), sizeof(TYPE) * size());
	}
	tmp = mat.T();
	tmp.copyRealise(true, true);
	if(CUDA){
		cuWrap::memD2D(ret.dataPrt() + size(), tmp.dataPrt(), sizeof(TYPE) * mat.size());
	}else{
		memcpy(ret.dataPrt() + size(), tmp.dataPrt(), sizeof(TYPE) * mat.size());
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::colJoint(const MatriX<TYPE, CUDA> & mat)  const{
		if(mat.rows()!= rows()){
		Assert("矩阵行数不一致，无法连接");
	}
	int rr = rows();
	int rc = cols() + mat.cols();
	MatriX<TYPE, CUDA> ret(rr, rc);
	MatriX<TYPE, CUDA> tmp(*this, ALL);
	if(CUDA){
		cuWrap::memD2D(ret.dataPrt(), tmp.dataPrt(), sizeof(TYPE) * size());
	}else{
		memcpy(ret.dataPrt(), tmp.dataPrt(), sizeof(TYPE) * size());
	}
	tmp = mat;
	tmp.copyRealise(true, true);
	if(CUDA){
		cuWrap::memD2D(ret.dataPrt() + size(), tmp.dataPrt(), sizeof(TYPE) * mat.size());
	}else{
		memcpy(ret.dataPrt() + size(), tmp.dataPrt(), sizeof(TYPE) * mat.size());
	}
	return ret;
};

template <typename TYPE, bool CUDA>
TYPE MatriX<TYPE,CUDA>::operator [](int idx) const{
	idx  = realPos(idx);
	TYPE ret;
	if(CUDA){		
		cuWrap::memD2H(&ret, dataPrt() + idx, sizeof(TYPE));
	}else{
		ret = dataPrt()[idx];
	}
	return ret * scale;
}
template <typename TYPE, bool CUDA>
TYPE MatriX<TYPE,CUDA>::operator ()(int rowIdx, int colIdx) const{
	return (*this)[elementPos(rowIdx, colIdx)];	
}