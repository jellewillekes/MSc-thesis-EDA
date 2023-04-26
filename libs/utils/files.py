import os


def get_file_extension(filename):
    filename, file_extension = os.path.splitext(filename)
    return file_extension


def get_file_name(filename):
    filename, file_extension = os.path.splitext(filename)
    return filename


def get_dir_name(path):
    return os.path.dirname(path)


def project_folder():
    """Root of the project"""
    lib_path = os.path.dirname(os.path.dirname(__file__))
    head, tail = os.path.split(lib_path)
    return head

