import subprocess
import time

def dir_files_to_list(dir):
  out=subprocess.Popen(["ls", dir],stdout=subprocess.PIPE).communicate()[0]
  return out.decode("utf-8").split('\n')[:-1]

def run_test(program):
    main_folder = "../TEST_CASES/TEST_INSTANCES"
    sub_folders = dir_files_to_list(main_folder)
    sub_folders.sort(key=(lambda x: int(x)))
    for f in sub_folders:
        arg=main_folder+"/"+f
        print(arg)
        start_time = time.time()
        res = subprocess.Popen([program, arg],stdout=subprocess.PIPE).communicate()[0]
        end_time = time.time()

        print(res)
        print('\n'+f+' '+str(end_time-start_time)+'\n\n')
run_test("./ESEGUI")