#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> 
#include <thrust/fill.h>
#include <thrust/replace.h> 
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
using namespace std;

#include "cuGlobal.h"
#include "deviceFuncs.h"
#include "cuWrap.h"
#include "cuRealise.h"



