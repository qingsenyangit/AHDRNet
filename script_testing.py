import time
import argparse
import torch.utils.data

from model import *
from running_func import *
from utils import *

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--trained_model_dir', default='./trained-model/')
parser.add_argument('--trained_model_filename', default='ahdr_model.pt')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()


torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

#load data
testimage_dataset = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image),
    batch_size=1)


#make folders of trained model and result
mk_dir(args.result_dir)


##
model = AHDR(args)
model.apply(weights_init_kaiming)
if args.use_cuda:
    model.cuda()

##
start_step = 0
# if args.load_model and len(os.listdir(args.trained_model_dir)):
model = model_load(model, args.trained_model_dir, args.trained_model_filename)

# In the testing, we need test on the whole image, so we defind a new variable
#  'Image_test_loaders' used to load the whole image
start = time.time()
loss = testing_fun(model, testimage_dataset, args)
end = time.time()
print('Running Time: {} seconds'.format(end - start))
