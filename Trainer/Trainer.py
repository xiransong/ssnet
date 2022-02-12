from .TrainTracer import TrainTracer
from Utils import io
from Utils.Timer import Timer
from Utils.utils import get_formatted_results

import numpy as np
import torch
import os.path as osp
from tqdm import tqdm


class Trainer:
    
    def __init__(self,
                 config,
                 model, loss_fn, opt, 
                 metrics,
                 train_dl, val_dl, test_dl,
                 ):
        '''
        requirements:
        
        config:
            results_root: 
            convergence_threshold: 
            val_freq: 
            epochs: 
            
        '''
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.opt = opt
        self.metrics = metrics
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.results_root = config['results_root']
        
        def save_best_model():
            self.model.save(root=self.results_root)
            
        def load_best_model():
            self.model.load(root=self.results_root)
        
        self.train_tracer = TrainTracer(convergence_threshold=config['convergence_threshold'],
                                        fn_save_best_model=save_best_model,
                                        record_root=self.results_root)
        self.load_best_model = load_best_model
        
        self.timer = Timer()
        
        self.epochs = config['epochs']
        self.val_freq = config['val_freq']
    
    def _train_loop(self):
        for epoch in range(self.epochs):
            
            if (epoch != 0) and (epoch % self.val_freq == 0):
                self.timer.start("val")
                with torch.no_grad():
                    print("epoch {} val...".format(epoch))
                    self.model.prepare_for_val()
                    
                    val_batch_outputs = []
                    for batch_data in tqdm(self.val_dl, desc="val"):
                        batch_output = self.model.val_or_test_a_batch(batch_data)
                        val_batch_outputs.append(batch_output)
                        
                    _, key_score, val_results = self.metrics(val_batch_outputs)
                    print("val:", val_results)
                    
                    val_results.update({"loss": np.nan if epoch == 0 else epoch_loss})
                    is_converged = self.train_tracer.check_and_save(key_score, epoch, val_results)
                    if is_converged:
                        break
                self.timer.end("val")

            epoch_loss = []
            self.model.prepare_for_train()
            
            print(">> epoch {}".format(epoch + 1))
            self.timer.start("epoch")
            for batch_data in self.train_dl:
                self.timer.start("batch")

                self.timer.start("batch_forward")
                output = self.model(batch_data)
                loss = self.loss_fn(output)
                self.timer.end("batch_forward")
                
                self.timer.start("batch_backward")
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                epoch_loss.append(loss.item())
                self.timer.end("batch_backward")
                
                self.timer.end("batch")
            self.timer.end("epoch")
            
            epoch_loss = np.mean(epoch_loss)
            print("loss {}".format(epoch_loss))
        
    def train(self):
        self.timer.start("train")
        
        try:
            self._train_loop()
        except KeyboardInterrupt:
            pass
        
        self.timer.end("train")
        self.timer.save_record(root=self.results_root)
        
    def test(self):
        print("test...")
        self.timer.start("test")
        with torch.no_grad():
            self.model.prepare_for_test()
            self.load_best_model()
            
            test_batch_outputs = []
            for batch_data in tqdm(self.test_dl, desc="test"):
                batch_output = self.model.val_or_test_a_batch(batch_data)
                test_batch_outputs.append(batch_output)
            
            S, _, test_results = self.metrics(test_batch_outputs)
            test_results['formatted'] = get_formatted_results(test_results)
            print("test:", test_results)
            
            io.save_json(osp.join(self.results_root, "test_results.json"), test_results)
            # io.save_pickle(osp.join(self.results_root, "S.pkl"), S)  # prediction scores
        self.timer.end("test")
        self.timer.save_record(root=self.results_root)
