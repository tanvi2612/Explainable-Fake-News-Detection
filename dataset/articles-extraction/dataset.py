# from warcio.archiveiterator import ArchiveIterator

# with open('CC-NEWS-20210219124957-00320.warc.gz', 'rb') as stream:
#     for record in ArchiveIterator(stream):
#         if record.rec_type == 'response':
#             print(record.rec_headers.get_header('WARC-Target-URI'))



def read_header(file_handler):
    header = {}
    line = next(file_handler)
    while line != '\n':
        key, value = line.split(': ', 1)
        header[key] = value.rstrip()
        line = next(file_handler)
    return header


def warc_records(path):
    with open(path) as fh:
        while True:
            line = next(fh)
            if line == 'WARC/1.0\n':
                output = read_header(fh)
                if 'WARC-Refers-To' not in output:
                    continue
                output["Content"] = next(fh)
                yield output



records = warc_records('CC-NEWS-20210219124957-00320.warc.gz')
next_record = next(records)

print(sorted(next_record.keys()))
