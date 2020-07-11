import os, shutil

java_files = []
Source = '/Users/liuxiaosu/codebase/kafka/'
Target = '/Users/liuxiaosu/codebase/kafka-bak/'

# Traverse all the files/dirs in the given path
def listdir(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path)
        elif os.path.splitext(file_path)[1]=='.java':
            java_files.append(file_path)

# Extract all the .java files from Source to Target
def makeJavaCollection():
    listdir(Source)
    if not os.path.exists(Target):
        os.mkdir(Target)
    for f in java_files:
        shutil.copy(f, Target)

if __name__=='__main__':

    makeJavaCollection()