import os

def mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)

def tuple_argument(s):
  return tuple(map(int, s.strip('()').split(',')))

def listdir(path):
  files = set()
  for f in set(os.listdir(path)):
    files.add(os.path.abspath(os.path.join(path, f)))
  return files

def write(filename, content):
  with open(filename, 'w') as f:
    f.write(content)

def convert_to_file_format(string):
  return ' '.join(word.lower() for word in string.split())
