import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import h5py
from utils import *
from config import cfg
from utils import load_data
from capsNet import CapsNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model, supervisor, num_label):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')
def evaluation1(model, supervisor, num_label):
    teX,file_names = load_test_dataset("./test1",cfg.image_size)
    f = h5py.File("dataset.hdf5","r")
    character_directory = f["character_names"][()]
    characters = []
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')
        for i in tqdm(range(0,teX.shape[0]//cfg.batch_size)):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            results = sess.run(model.softmax_v, {model.X: teX[start:end]})
            for j in range(results.shape[0]):
                result = results[j].reshape((100,))
                max_index = np.argpartition(result, -5)[-5:]
                char = ""
                for t in range(0,5):
                    char+=character_directory[max_index[t]]
                characters.append(char)
        fd_results = open("results.csv","w")
        fd_results.write("filename,label\n")
        for i in range(0,teX.shape[0]):
            fd_results.write(file_names[i]+ ','+characters[i]+"\n")
        fd_results.flush()
        fd_results.close()

def main(_):
    tf.logging.info(' Loading Graph...')

    model = CapsNet()
    tf.logging.info(' Graph loaded')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model, sv, cfg.category)
        tf.logging.info('Training done')
    else:
        evaluation1(model, sv, cfg.category)

if __name__ == "__main__":
    tf.app.run()
