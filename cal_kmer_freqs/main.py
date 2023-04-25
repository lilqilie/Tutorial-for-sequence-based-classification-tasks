import cal


# 从fasta目录下，合并所有fasta文件
# 保留每个文件中第一个菌的名字，接着拼接序列
# 合并后的文件目录传入计算kmer频率函数，输出kmer频率
def config_set():
    config = {
        # 1. 待合并的fasta目录
        'fasta_path': r'',

        # 合并后的文件输出目录
        'combined_file': 'patho_all.fna',

        # 2.cal kmer freqs
        # cpu 核数
        'num_procs': 20,
        # 需要计算kmer频率的大小
        'ks': [3, 4, 5, 6, 7],
        # 设置最终计算的kmer频率输出文件目录
        'freqs_file': 'output/patho_all.csv',
    }
    return config


if __name__ == '__main__':
    config = config_set()
    # 1. 合并序列
    combine_fna.combine(config['fasta_path'], config['combined_file'])
    # 2. get fasta name
    fasta_name = config['nopatho']
    seq_lens = get_name(fasta_name)
    print('nonpathogen numbers:', len(seq_lens))
    print('nonpathogen all length:', sum(seq_lens))
    # 3. 计算kmer频率函数 
    cal.cal_main(config)
