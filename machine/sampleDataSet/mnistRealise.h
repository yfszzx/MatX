mnist::mnist(string path){		
	int t;
	unsigned char *tmp;
	cout<<"\n"<<"正在载入数据集....";
	cout<<path<<"train-images.idx3-ubyte";
	//数据
	ifstream fin(path+"train-images.idx3-ubyte", ios::binary); 	
	if(!fin){
		cout<<"路径错误";
		char s;
		cin>>s;
		return;
	}

	fin.read((char *)&t,sizeof(int));//magic_num;
	fin.read((char *)&train_num,sizeof(int));//data_num		
	fin.read((char *)&rows,sizeof(int));
	fin.read((char *)&columns,sizeof(int));
	rows=28;
	columns=28;
	train_num=60000;
	input_num=rows*columns;
	output_num=10;
	cout<<"\n"<<"行数:"<<rows<<" 列数:"<<columns<<endl<<"训练集数量:"<<train_num;
	tmp=new unsigned char[rows*columns*train_num*sizeof(char)];
	fin.read((char *)tmp,rows*columns*train_num*sizeof(char));

	data=new float[rows*columns*train_num*sizeof(float)];
	for(int i=0;i<rows*columns*train_num;i++){

		data[i]=float(tmp[i])/255;
	}
	input=new float *[train_num];
	for(int i=0;i<train_num;i++)
		input[i]=data+rows*columns*i;
	delete [] tmp;
	fin.close();
	//标签
	fin.open(path+"train-labels.idx1-ubyte",ios::binary); 	
	if(!fin){
		cout<<"路径错误";
		char s;
		cin>>s;
		return;
	}
	fin.read((char *)&t,sizeof(int));//magic_num;
	fin.read((char *)&t,sizeof(int));//data_num	
	tmp=new unsigned char[train_num*sizeof(char)];
	fin.read((char *)tmp,train_num*sizeof(char));
	label_d=new float[train_num*output_num*sizeof(float)];		
	label=new float *[train_num*sizeof(float)];
	for(int i=0;i<train_num;i++){			
		label[i]=label_d+i*output_num;
		for(int j=0;j<output_num;j++){
			label[i][j]=(j==tmp[i])?1:0;
		}

	}
	delete [] tmp;	
	fin.close();



	//测试集
	//数据
	fin.open(path+"t10k-images.idx3-ubyte", ios::binary); 		
	fin.read((char *)&t,sizeof(int));//magic_num;
	fin.read((char *)&test_num,sizeof(int));//data_num
	fin.read((char *)&t,sizeof(int));
	fin.read((char *)&t,sizeof(int));
	test_num=10000;
	cout<<"测试集数量:"<<test_num;
	tmp=new unsigned char[rows*columns*test_num*sizeof(char)];
	fin.read((char *)tmp,rows*columns*test_num*sizeof(char));		
	data_t=new float[rows*columns*test_num*sizeof(float)];
	for(int i=0;i<rows*columns*test_num;i++)
		data_t[i]=float(tmp[i])/255;
	input_t=new float *[test_num];
	for(int i=0;i<test_num;i++)
		input_t[i]=data_t+rows*columns*i;
	delete [] tmp;
	fin.close();
	//标签
	fin.open(path+"t10k-labels.idx1-ubyte", ios::binary); 		
	fin.read((char *)&t,sizeof(int));//magic_num;
	fin.read((char *)&t,sizeof(int));//data_num		
	tmp=new unsigned char[test_num*sizeof(char)];
	fin.read((char *)tmp,test_num*sizeof(char));
	label_t_d=new float[test_num*output_num*sizeof(float)];		
	label_t=new float *[test_num*sizeof(float)];
	for(int i=0;i<test_num;i++){			
		label_t[i]=label_t_d+i*output_num;
		for(int j=0;j<output_num;j++){
			label_t[i][j]=(j==tmp[i])?1:0;
		}
	}
	delete [] tmp;
	fin.close();
}

void mnist::showImg(int idx,char type,bool label_show){
	float **&ipt=(type=='i')?input:input_t;
	//float **&lbl=(type=='i')?label:label_t;
	cout<<"\n";
	for(int j=0;j<columns+2;j++)cout<<"-";
	cout<<"\n";
	for(int i=0;i<rows;i++){
		cout<<"|";
		for(int j=0;j<columns;j++){
			if(ipt[idx][i*rows+j]<0.25){
				cout<<" ";
				continue;
			}
			if(ipt[idx][i*rows+j]<0.5){
				cout<<".";
				continue;
			}
			if(ipt[idx][i*rows+j]<0.75){
				cout<<"+";
				continue;
			}
			cout<<"*";
		}
		cout<<"|\n";
	}		
	for(int j=0;j<columns+2;j++)cout<<"-";
	if(label_show)cout<<"\n"<<"序号:"<<idx<<" 数字:"<<value(idx,type);
	cout<<endl;
}	
int mnist::input_idx(char type){
	int idx;
	cout<<"\n"<<"输入序号(-1返回):";
	cin>>idx;
	if(idx<0)return -1;
	if(type=='i'){
		if(idx>=train_num){
			cout<<"\n"<<"【错误】序号太大";
			return -2;
		}
	}else{
		if(idx>=test_num){
			cout<<"\n"<<"【错误】序号太大";
			return -2;
		}
	}
	return idx;
}
void mnist::check(){
	int idx;
	char type;
	cout<<"\n";
	cout<<"\n"<<"\t<查看mnist图像>";
	cout<<"\n"<<"选择数据集[i.训练集 t.测试集]:";
	cin>>type;
	do{
		idx=input_idx(type);
		if(idx==-1)break;
		if(idx==-2)continue;
		showImg(idx,type);
	}while(1);
}
int mnist::value(float *tgt){
	for(int i=0;i<output_num;i++)	if(tgt[i]>0)return i;
	return -1;
}
int mnist::value(int idx,char mod){
	if(mod=='i')return value(label[idx]);
	else return value(label_t[idx]);
}
int mnist::get_out_value(float *out){
	float mx=-10;int ret=-1;
	for(int i=0;i<output_num;i++){
		if(out[i]>mx){
			mx=out[i];
			ret=i;
		}
	}
	return ret;
}
void mnist::show_out(float *out){
	cout<<"result:";
	int val=get_out_value(out);
	cout<<val<<endl;		
	for(int i=0;i<output_num;i++)
		cout<<" "<<i<<":"<<out[i];
}
float mnist::accuracy(float *out, float *tgt, int num){
	int count=0;	
	for(int i=0;i<num;i++){
		int v=value(tgt+i*output_num);
		int o=get_out_value(out+i*output_num);
		if(v==o)count++;			
	}
	float ret=float(count)/num;
	cout<<" 正确率:"<<(ret*100)<<"%("<<num<<"个样本)";
	return ret;
}
void mnist::show_wrong(float *out,char mod){
	int count=0;
	int num=(mod=='t')?train_num:test_num;
	for(int i=0;i<num;i++){
		int v=value(i,mod);
		int o=get_out_value(out+output_num*i);
		if(v==o)count++;
		else{
			showImg(i,mod);
			show_out(out+output_num*i);
			getchar();
		}
	}

}
