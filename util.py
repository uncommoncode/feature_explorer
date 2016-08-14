import fnmatch
import hashlib
import os

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def stable_pseudo_shuffle(strings):
    hashed_strings = []
    for string in strings:
        md5 = hashlib.md5()
        md5.update(string)
        hashed_strings.append((md5.hexdigest(), string))
    return map(lambda t: t[1], sorted(hashed_strings, key=lambda t: t[0]))

def iter_chunks(sequence, chunk_size):
    return (sequence[offset:offset + chunk_size] for offset in xrange(0, len(sequence), chunk_size))
