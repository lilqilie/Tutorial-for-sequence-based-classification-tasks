import os
import time
import urllib
from multiprocessing import Pool, Manager

import pandas as pd
import wget

from download_config import Config

config = Config()


def ftp_download(args):
    # 将需要的下载的url复制下来，左边的URL为变量，可根据实际情况填写其他的
    name = args[0]
    fna_file_url = args[1]
    output_file_name = args[2]
    successful_list = args[3]
    e = ''
    if os.path.exists(output_file_name):
        print('output path exists ', output_file_name, ' pass')
        successful_list.append(1)
        print('downloaded:', len(successful_list))
        return

    start = time.time()
    for i in range(config.retry_times):
        print('downloading ', fna_file_url, 'times:', str(i + 1) + '/' + str(config.retry_times))
        try:
            do = wget.download(fna_file_url, config.output_path)
        except urllib.error.URLError as e:
            print('error:', e)
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(now, ' download ' + name + ' fail, retrying...')
            continue
        if do and os.path.exists(output_file_name):
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(now, ' download ' + fna_file_url + ' successfully')
            print(fna_file_url, ' cost:', time.time() - start, 's')
            successful_list.append(1)
            print('downloaded:', len(successful_list))

            break
        else:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(now, ' download ' + fna_file_url + ' fail, retrying...')
            continue
    #
    if not os.path.exists(output_file_name):
        if e:
            print('cant download ', name, ' reason:', e)
        f = open(config.fail_log_path, 'a')
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if e:
            f.write(now + ',' + fna_file_url + ',' + str(e))
            f.write('\n')
            f.close()
        else:
            f.write(now + ',' + fna_file_url)
            f.write('\n')
            f.close()


def multi_download(args):
    with Pool(config.cpu_worker_num) as p:
        p.map(ftp_download, args)


def main():
    successful_list = Manager().list()
    df = pd.read_csv(config.input_path)
    df_all = pd.read_csv('Tao_NCBI_22June_RefSeq_32927_Complete_1NP_2P.csv')
    ftp_list = df[config.csv_col].to_list()
    print('download list numbers:', len(ftp_list))
    args = []
    for ftp in ftp_list:
        raw_name = ftp.split('/')[-1][:-4]
        real_ftp = df_all[df_all['assembly_accession'] == raw_name]['ftp_path'].to_list()[0]
        real_name = real_ftp.split('/')[-1]
        fna_file_url = real_ftp + '/' + real_name + config.file_type
        output_file_name = config.output_path + real_name + config.file_type
        args.append([real_name, fna_file_url, output_file_name, successful_list])
    multi_download(args)
    # ftp_download(args[11111])


if __name__ == '__main__':
    main()
