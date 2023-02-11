from experiments.qa_gnn_eraser import run,erase,link_dataset
import datetime

if __name__ == "__main__":
    dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

    linked_train = link_dataset('train')
    linked_dev = link_dataset('dev')
    linked_test = link_dataset('test')

    erase('train', dt) 
    erase('dev', dt) 
    erase('test', dt) 

    run('train',dt)
    run('dev',dt)
    run('test',dt)

    print("DONE")