# Extract all the .java files from Source to Target
import os, shutil

Source = '/Users/liuxiaosu/codebase/hadoop/'
Target = '/Users/liuxiaosu/codebase/hadoop-bak/'

# Traverse all the files/dirs in the given path
files = []
def listdir(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path)
        elif os.path.splitext(file_path)[1]=='.java':
            files.append(file_path)

def makeJavaCollection():
    if not os.path.exists(Target):
        os.mkdir(Target)
    listdir(Source)
    for f in files:
        shutil.copy(f, Target)

if __name__=='__main__':

    makeJavaCollection()