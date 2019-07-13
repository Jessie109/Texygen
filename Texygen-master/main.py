import getopt
import sys
import tensorflow as tf

from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


def set_gan(gan_name, size):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['mle'] = Mle
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 569
        gan.emb_dim = 32
        gan.hidden_dim = 32
        gan.sequence_length = 20
        gan.filter_size = [2, 3, 4]
        gan.num_filters = [20, 20, 20]
        gan.l2_reg_lambda = 0.2
        gan.dropout_keep_prob = 0.75
        gan.batch_size = 30
        gan.generate_num = 200
        gan.start_token = 0
        gan.gen_file_data_itr = 'gen_data/seqgan/coco/'
        gan.pre_epoch_num = 10
        gan.adversarial_epoch_num = 10

        gan.oracle_file = 'save/seqgan/test/oracle.txt'
        gan.generator_file = 'save/seqgan/test/generator.txt'
        gan.test_file = 'save/seqgan/test/test_file.txt'

        
        if(gan_name.lower()=='leakgan'):
#             flags = tf.app.flags
#             FLAGS = flags.FLAGS
#             flags.DEFINE_boolean('restore', False, 'Training or testing a model')
#             flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
#             flags.DEFINE_integer('length', 20, 'The length of toy data')
#             flags.DEFINE_string('model', "", 'Model NAME')
            gan.dis_embedding_dim = 300
            gan.goal_size = 16
            
#         if(gan_name.lower()=='gsgan'):
#             gan.pre_epoch_num = 0
#         if(gan_name.lower()=='leakgan'):
#             flags = tf.app.flags
#         if(gan_name.lower()=='leakgan'):
#             flags = tf.app.flags          

        
        
        
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)



def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def parse_cmd(argv):
    
    try:
        opts, args = getopt.getopt(argv, "hg:t:d:s:i:")

        opt_arg = dict(opts)
        print('opt_arg',opt_arg)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <gan_type>')
            print('       python main.py -g <gan_type> -t <train_type>')
            print('       python main.py -g <gan_type> -t realdata -d <your_data_location>')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use MLE training only...')
            gan = set_gan('mle')
        else:
            gan = set_gan(opt_arg['-g'], opt_arg['-s'])
        if not '-t' in opt_arg.keys():
            gan.train_oracle()
        else:
            gan_func = set_training(gan, opt_arg['-t'])
            if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
                size = int(opt_arg['-s'])
                itr = int(opt_arg['-i'])
                gan_func(size,itr, opt_arg['-d'])
            else:
                gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])
