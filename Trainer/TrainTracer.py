import os.path as osp


class TrainTracer:
    
    def __init__(self, convergence_threshold, 
                 record_root, fn_save_best_model=None, record_filename="train_record.txt"):
        self.fn_save_best_model = fn_save_best_model
        self.convergence_threshold = convergence_threshold
        self.record_file = osp.join(record_root, record_filename)
        
        self.best_key_score = None
        self.best_epoch = None
        
    def check_and_save(self, key_score, epoch, val_results: dict, fn_save_best_model=None):
        val_results['epoch'] = epoch
        
        if self.best_epoch is None:
            with open(self.record_file, "w") as f:
                f.write(','.join(val_results.keys()) + '\n')
                
        with open(self.record_file, "a") as f:
            f.write(
                ','.join(map(
                    lambda x: "{:.4g}".format(x) if isinstance(x, float) else str(x), 
                    val_results.values())
                ) + '\n'
            )
        
        if self.best_key_score is None or key_score > self.best_key_score:
            print(">> new best score:", key_score)
            self.best_key_score = key_score
            self.best_epoch = epoch
            if fn_save_best_model is not None:
                fn_save_best_model()
            else:
                self.fn_save_best_model()
        else:
            print(">> distance between best_epoch:", epoch - self.best_epoch, "threshold:", self.convergence_threshold)
            
        is_converged = False
        if epoch - self.best_epoch >= self.convergence_threshold:
            print("converged at epoch {}".format(epoch))
            is_converged = True

        return is_converged
