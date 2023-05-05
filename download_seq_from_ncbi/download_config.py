class Config:
    def __init__(self):
        self.input_path = 'NCBI_22June_RefSeq_32927_Complete_1NP_2P_taxnonmy.csv'
        self.output_path = ''
        self.cpu_worker_num = 32
        # file suffix of final url
        self.file_type = '_genomic.fna.gz'
        self.csv_col = 'ftp_path'

        # times of re-downloads
        self.retry_times = 10
        self.fail_log_path = ''
