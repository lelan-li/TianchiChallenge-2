import os
os.environ['PYTHONPATH'] = '%s:%s' % ('/home/caffe/python', '/workspace/pai')
import sys
sys.path.append('/home/caffe/python')
sys.path.append('/workspace/pai')
import caffe
import layer


# breakpoint train
weights = '/workspace/pai/ali_challenge2/1_train/caffe_model/7_iter_600.caffemodel'
solver = caffe.get_solver("/workspace/pai/ali_challenge2/1_train/solver.prototxt")
solver.net.copy_from(weights)
solver.solve()

# train
# solver = caffe.SGDSolver("/workspace/pai/ali_challenge2/1_train/solver.prototxt")
# solver.solve()

# console train
# cmd1 = "export $PYTHONPATH=/workspace/pai/ali_challenge2/1_train/layer:$PYTHONPATH"
# cmd2 = "/home/caffe/bin/caffe.bin train --solver='/workspace/pai/ali_challenge2/1_train/solver.prototxt'" \
#       " --snapshot='/workspace/pai/ali_challenge2/1_train/caffe_model/0caffe_model_iter_8000.solverstate'"
# os.system(cmd1)
# os.system(cmd2)
