import os, shutil

Source = '/Users/liuxiaosu/codebase/kafka/'
Target = '/Users/liuxiaosu/codebase/kafka-bak/'

# Traverse all the files/dirs in the given path
def listdir(path):
    files = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path)
        elif os.path.splitext(file_path)[1]=='.java':
            files.append(file_path)
    return files

# Extract all the .java files from Source to Target
def makeJavaCollection():
    if not os.path.exists(Target):
        os.mkdir(Target)
    for f in listdir(Source):
        shutil.copy(f, Target)

if __name__=='__main__':

    makeJavaCollection()