from experiments.qa_gnn_eraser import run
import datetime

if __name__ == "__main__":
    dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S') 
    run('dev',dt)
    run('train',dt)
    run('test',dt)
    print('DONE')