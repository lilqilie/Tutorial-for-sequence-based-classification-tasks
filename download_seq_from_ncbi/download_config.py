class Config:
    def __init__(self):
        self.input_path = ''
        self.output_path = ''
        self.cpu_worker_num = 32
        # file suffix of final url
        self.file_type = '_genomic.fna.gz'
        self.csv_col = 'urls'

        # times of re-downloads
        self.retry_times = 10
        self.fail_log_path = ''
