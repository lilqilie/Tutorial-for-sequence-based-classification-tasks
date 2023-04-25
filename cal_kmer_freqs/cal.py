import itertools
import multiprocessing as mp
import time
import warnings
from multiprocessing import Manager
import numpy as np
from tqdm import tqdm
from cal_utils import readfq, mer2bits, get_rc, count_kmers


warnings.filterwarnings("ignore")


def compute_kmer_inds(ks):
    ''' Get the indeces of each canonical kmer in the kmer count vectors
    '''
    kmer_list = []
    kmer_inds = {k: {} for k in ks}
    kmer_count_lens = {k: 0 for k in ks}

    alphabet = 'ACGT'
    for k in ks:
        all_kmers = [''.join(kmer)
                     for kmer in itertools.product(alphabet, repeat=k)]
        all_kmers.sort()
        ind = 0
        for kmer in all_kmers:
            bit_mer = mer2bits(kmer)
            rc_bit_mer = mer2bits(get_rc(kmer))
            if rc_bit_mer in kmer_inds[k]:
                kmer_inds[k][bit_mer] = kmer_inds[k][rc_bit_mer]
            else:
                kmer_list.append(kmer)
                kmer_inds[k][bit_mer] = ind
                kmer_count_lens[k] += 1
                ind += 1
    return kmer_inds, kmer_count_lens, kmer_list


def get_seq_lengths(infile):
    ''' Read in all the fasta entries,
        return arrays of the headers, and sequence lengths
    '''
    sequence_names = []
    sequence_lengths = []
    seqs = []
    fp = open(infile)
    for name, seq, _ in readfq(fp):
        sequence_names.append(name)
        sequence_lengths.append(len(seq))
        seqs.append(seq)
    fp.close()

    # print('len of sequence_lengths', len(sequence_names), len(sequence_lengths))

    return sequence_names, sequence_lengths, seqs


def get_seq(infile):
    seq_names = []
    seqs = []
    fp = open(infile)
    i = 0

    for name, seq, _ in readfq(fp):
        seq_names.append(name)
        seqs.append(seq)
        i += 1
    print("Read {} sequences".format(i))
    return seq_names, seqs


def get_num_frags(seq_lengths, length, coverage=5):
    ''' Compute how many fragments to generate
    '''
    # filter out sequences that are significantly shorter than the length ���˵����Զ����������е�����
    # filtered_seq_lengths = [l for l in seq_lengths if l > 0.85*length]
    # filtered_seq_lengths =
    tot_seq_len = sum(seq_lengths)
    num_frags_for_cov = int(np.ceil(tot_seq_len * coverage / float(length)))
    num_frags = min(90000, num_frags_for_cov)
    # num_frags = (tot_seq_len)
    return num_frags


# def get_start_inds(seq_names, seq_lengths, num_frags, length):
def get_start_inds(seq_names):
    ''' Randomly simulate fragments of a specific length from the sequences ���ģ���������ض����ȵ�Ƭ��
    '''
    # filter out sequences that are significantly shorter than the length
    # filtered_seq_names = [seq_names[i] for i,v in enumerate(seq_lengths) if v > 0.05*length]
    # filtered_seq_lengths = [l for l in seq_lengths if l > 0.05*length]
    # filtered_seq_lengths = seq_lengths
    # tot_seq_len = sum(seq_lengths)
    # length_fractions = [float(l) / float(tot_seq_len) for l in seq_lengths]
    inds_dict = {}
    for name in seq_names:
        inds_dict[name] = [0]
    return inds_dict


def get_seqs(infile, inds_dict, l):
    ''' Create array of the sequences
    '''
    seqs = []
    fp = open(infile)
    for name, seq, _ in readfq(fp):
        seqs.append(seq)
    fp.close()
    return seqs


def cal_kmer_freqs(fasta_file, num_procs, ks):
    time_start = time.time()
    patho_names, patho_lengths, patho_seqs = get_seq_lengths(fasta_file)
    # for l in lens:
    # coverage=1 # TODO: make this command line option
    kmer_inds, kmer_count_lens, kmer_list = compute_kmer_inds(ks)
    print('kmer index calculated!')
    print('start cal kmer...')
    # print('kmer_inds: /n',kmer_inds)
    # print('kmer_count_lens:/n', kmer_count_lens)
    # pool = mp.Pool(num_procs)
    patho_list = Manager().list()
    for cur in np.arange(len(patho_seqs)):
        patho_list.append(0)
    pbar = tqdm(patho_seqs)
    # pool.map(count_kmers, [[ind, s, ks, kmer_inds, kmer_count_lens, patho_list] \
    #                        for ind, s in enumerate(pbar)])
    params_list = []
    for ind, s in enumerate(patho_seqs):
        # pbar.set_description('Processing ' + ind)
        params_list.append(
            [ind, s, ks, kmer_inds, kmer_count_lens, patho_list])

    with mp.Pool(num_procs) as p:
        print(list((tqdm(p.imap(count_kmers, params_list),
              total=len(patho_list), desc='nums'))))
    patho_freqs = np.array(patho_list)
    time_end = time.time()
    print('calculating costs:', int(time_end - time_start), 's')

    return patho_freqs


def cal_main(config):
    print('Calculating kmer frequency...')
    freqs = cal_kmer_freqs(config['combined_file'],
                           config['num_procs'], config['ks'])
    np.savetxt(config['freqs_file'], freqs, delimiter=",")
