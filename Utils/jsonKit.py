import json


def dumpJson(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def loadJson(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data
