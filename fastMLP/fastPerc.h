class perceptrons{
private:
	float *tmp_array;	
	int len;
	void free_mem();
	int block_y;
	float alpha;
	float beta;
	float a_alpha;
public:
	int input_dimen;
	int nodes_num;
	char type;//l:line,t:tanh,s:sigmoid
	int data_num;
	int weight_len; 
	float *weight;
	float *output;	
	float *deriv;
	float decay;
	char decay_mod;
	perceptrons();
	~perceptrons();
	void init_struct();
	void init_struct(int dimen,int nodes,char tp);
	void show_struct(int idx);
	int memery();
	void set_data_num(int num);
	void run(float *input);
	void get_deriv(float *input);
	void get_sub_deriv(float *sub_deriv,char sub_type);
	float weight_decay_sum();
	void get_decay_deriv();
};
void perceptrons::run(float *input){
	CUBT(cublasSgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,
		nodes_num,data_num,input_dimen, &alpha,weight,nodes_num,input,input_dimen,&beta,output,nodes_num));
	dim3 blk(nodes_num,block_y);
	g_add_threshold<<<blk,g_threads>>>(weight+input_dimen*nodes_num,output,nodes_num,data_num);		
	CUDA_CHECK;
	if(type=='t')g_tanh(output,nodes_num*data_num);
	if(type=='s')g_sigmoid(output,nodes_num*data_num);	
	//coutd<<"oo"<<array_length(output,nodes_num*data_num);
}
