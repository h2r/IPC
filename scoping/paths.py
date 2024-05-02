import os

library_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(library_dir)


def get_effectively_relevant_fluents_file_path(sas_file):
    basename, ext = os.path.splitext(sas_file)
    erf_file = basename + "_erf" + ".txt"
    return erf_file


def get_scoped_file_path(unscoped_file):
    return add_path_suffix(unscoped_file, "_scoped")


def add_path_suffix(p, s):
    basename, ext = os.path.splitext(p)
    return basename + s + ext


def replace_extension(p, s):
    basename, ext = os.path.splitext(p)
    return basename + "." + s
