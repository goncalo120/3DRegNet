# main.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:


from config import get_config, print_usage
from network import Network
from data.data import load_data

def main(config):

    #Initialize Network
    net = Network(config)

    data = {}

    if config.run_mode == 'train':

        data['train'] = load_data(config, 'train')

        net.train(data)

    if config.run_mode == 'test':

        data['test'] = load_data(config, 'test')

        net.test(data)

    return 0

if __name__ == "__main__":

    config, unparsed = get_config()

    if len(unparsed) > 0:
        # print_usage()
        exit(1)

    main(config)
