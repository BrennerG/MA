from experiments.qa_gnn_eraser import run
from experiments.qa_gnn_eraser import erase
import datetime

if __name__ == "__main__":
    dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    run('train',dt)
    run('dev',dt)
    run('test',dt)