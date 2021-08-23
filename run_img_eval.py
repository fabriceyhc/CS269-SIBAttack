import os
import sys
import numpy as np

import subprocess as sp

def main():

    # all eval variables
    datasets = ['MNIST', 'CIFAR10']
    augs = ['basicaug', 'randaug', 'mixup2', 'cutmix2', 'tile']
    target_probs = np.linspace(start=0, stop=1, num=5)
    resize_probs = np.linspace(start=0, stop=1, num=5)
    num_tiles = [4, 9]

    # generate cmd strings
    cmds = []
    for ds in datasets:
        for aug in augs: 
            cmd = "python -u train.py"
            cmd += " --dataset " + ds
            # skip these combinations of dataset and aug  
            if (("MNIST" in ds and "randaug" in aug) or ("CIFAR10" in ds and "basicaug" in aug)):
                continue
            cmd += ' --use_' + aug
            if aug in ['basicaug', 'randaug']:
                out = ' --outdir results/' + ds + '/' + aug 
                cmds.append(cmd + out)
                continue
            for target_prob in target_probs:   
                tp = ' --target_prob ' + str(target_prob)   
                if target_prob > 0:
                    tp += ' --update_targets' 
                if 'cutmix2' in aug:  
                    for resize_prob in resize_probs:
                        rp  = ' --resize_prob ' + str(resize_prob)
                        out = ' --outdir results/' + ds + '/' + aug + '/tp_' + str(target_prob) + '_rp_' + str(resize_prob)
                        cmds.append(cmd + tp + rp + out)
                elif 'tile' in aug:
                    for num_tile in num_tiles:
                        nt = ' --num_tiles ' + str(num_tile)
                        out = ' --outdir results/' + ds + '/' + aug + '/tp_' + str(target_prob) + '_nt_' + str(num_tile)
                        bs = ' --batch_size 64'
                        cmds.append(cmd + tp + nt + out + bs)   
                else:
                    out = ' --outdir results/' + ds + '/' + aug + '/tp_' + str(target_prob) 
                    cmds.append(cmd + tp + out)   

    print(cmds)  

    # run all cmds one after the other
    for cmd in cmds:
        p = sp.Popen(cmd, stdout=sp.PIPE)
        out, err = p.communicate()
        result = out.decode().split('\n')
        for line in result:
            if not line.startswith('#'):
                print(line)

if __name__ == '__main__':
    main()