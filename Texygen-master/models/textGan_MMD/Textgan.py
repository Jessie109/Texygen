from time import time

from models.Gan import Gan
from models.textGan_MMD.TextganDataLoader import DataLoader, DisDataloader
from models.textGan_MMD.TextganDiscriminator import Discriminator
from models.textGan_MMD.TextganGenerator import Generator
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


class TextganMmd(Gan):
    def __init__(self, oracle=None):
        super(TextganMmd, self).__init__()
        # you can change parameters, generator here
        self.vocab_size = 409
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 36
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 20
        self.generate_num = 40
        self.start_token = 0
        
        self.pre_epoch_num = 50
        self.adversarial_epoch_num = 80

        self.temps = [0.001, 0.5, 1.0, 1.5, 2.0]
        self.generate_temp_size = 200

        self.name = 'textgan'
        self.generate_temp_dataset = 'emnlp'

        self.oracle_file = 'save/textgan/oracle.txt'
        self.generator_file = 'save/textgan/generator.txt'
        self.test_file = 'save/textgan/test_file.txt'

        self.real_file = ''
        self.result_file = 'save/result_file.txt'

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        g_embeddings = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def init_metric(self):

        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

#         from utils.metrics.DocEmbSim import DocEmbSim
#         docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
#         self.add_metric(docsim)

    def train_discriminator(self):
        for _ in range(3):
            x_batch, z_h = self.generator.generate(self.sess, True)
            y_batch = self.gen_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
                self.discriminator.zh: z_h,
                self.discriminator.input_x_lable: [[1, 0] for _ in x_batch],
                self.discriminator.input_y_lable: [[0, 1] for _ in y_batch],
            }
            _ = self.sess.run(self.discriminator.train_op, feed)

    def train_generator(self):
        z_h0 = np.random.uniform(low=-.01, high=.01, size=[self.batch_size, self.emb_dim])
        z_c0 = np.zeros(shape=[self.batch_size, self.emb_dim])

        y_batch = self.gen_data_loader.next_batch()
        feed = {
            self.generator.h_0: z_h0,
            self.generator.c_0: z_c0,
            self.generator.y: y_batch,
        }
        _ = self.sess.run(fetches=self.generator.g_updates, feed_dict=feed)
        pass

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super(TextganMmd, self).evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super(TextganMmd, self).evaluate()

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        self.log = open('experiment-log-textgan.csv', 'w')
        oracle_code = generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        del oracle_code
        print('adversarial training:')
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(100):
                self.train_generator()
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        from utils.oracle.OracleCfg import OracleCfg
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1
        g_embeddings = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        from utils.metrics.Cfg import Cfg
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        import json
        from utils.text_process import get_tokenlized
        from utils.text_process import code_to_text
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.global_variables_initializer())

        self.log = open('experiment-log-textgan-cfg.csv', 'w')
        oracle_code = generate_samples(self.sess, self.generator, self.batch_size, self.generate_num,
                                       self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')

        del oracle_code
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for i in range(100):
                self.train_generator()
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()
        return

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)

        g_embeddings = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict

    def init_real_metric(self):
#         from utils.metrics.DocEmbSim import DocEmbSim
#         docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
#         self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)


    def train_real(self, size,itr, data_loc=None):
        sizes = [200, 400, 600, 800, 1000]
        # size_files = ['1_200.txt','2_200.txt','3_200.txt']
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized

        data_loc = 'train_data/coco/'+str(itr)+'_'+str(size)+'.txt'
        # data_loc = 'train/'+size_files[cv_itr]
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
#         print(wi_dict)
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        def get_real_test_file_temp(write_file, dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(write_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        def get_real_code():
            text = get_tokenlized(self.oracle_file)

            def toint_list(x):
                return list(map(int, x))

            codes = list(map(toint_list, text))
            return codes

        self.sess.run(tf.global_variables_initializer())

        self.log = open('experiment-log-textgan-real.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                # write generate.txt in indexes of the gen sentences
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                # write ^ in words
                get_real_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()
        oracle_code = get_real_code()


        print('adversarial training:')
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(100):
                self.train_generator()
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()

        # bleu = Bleu( self.test_file, self.real_file, self.sess)
        # sbleu = SelfBleu(self.test_file, self.sess)
        # scorefile = open(self.result_file, 'a+')

        for alpha in self.temps:
            gen_file= 'gen_data/textgan/coco/'+str(itr)+'_'+str(size)+'_'+str(alpha)+'.txt'
#                     gen_file= 'gen_data/seqgan/coco/'+'ds.txt'
            print('alpha', alpha, gen_file)
            generate_samples_temp(self.sess, self.generator, self.batch_size, self.generate_num, alpha, self.generator_file)
            get_real_test_file_temp(gen_file)
            # scores = self.evaluate_test_with_temp(bleu,sbleu )
            # scorefile.write(str(alpha)+'##'+scores[0]+'##'+scores[1])

