import json
import pandas as pd
import zipfile

from libs.utils import files


def unzip_data():
    with zipfile.ZipFile(f'{files.project_folder()}/data.zip', 'r') as ref:
        ref.extractall(f'{files.project_folder()}')
    pass


def daycounts():
    data = pd.read_csv(f'{files.project_folder()}/data/avro-daycounts.csv')
    data = data.set_index(['day'])
    data.index = pd.to_datetime(data.index)

    return data


def issues_csv():
    data = pd.read_csv(f'{files.project_folder()}/data/avro-issues.csv')

    return data


def issues_json():

    data = [json.loads(line) for line in open(f'{files.project_folder()}/data/avro-issues.json', 'r')]

    return data


def transitions():
    data = pd.read_csv(f'{files.project_folder()}/data/avro-transitions.csv')

    return data


def formatted_issue():
    with open(f'{files.project_folder()}/data/formatted-issue.json') as json_file:
        data = json.load(json_file)

    return data
