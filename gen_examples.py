import random

import exrex
import os

POS_EXAMPLES_NAME = 'pos_examples.txt'
NEG_EXAMPLES_NAME = 'neg_examples.txt'
TEST_EXAMPLES_NAME = 'test_examples.txt'
TRAIN_EXAMPLES_NAME = 'train_examples.txt'

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

POS_REGEX = '[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'

EXAMPLES = 500


def main():
    generate_examples('pos')
    generate_examples('neg')
    generate_examples('test')
    merge_train_data()


def generate_examples(type):
    examples_file_name = POS_EXAMPLES_NAME
    if type == 'neg':
        examples_file_name = NEG_EXAMPLES_NAME
    if type == 'test':
        examples_file_name = TEST_EXAMPLES_NAME

    examples_file = open(os.path.join(data_dir, examples_file_name), 'w')

    type_inner = type
    for i in range(0, EXAMPLES):
        if type == 'test':
            if i < EXAMPLES / 2:
                type_inner = 'pos'
            else:
                type_inner = 'neg'

        if type_inner == 'pos':
            example = exrex.getone(POS_REGEX)
        elif type_inner == 'neg':
            example = exrex.getone(POS_REGEX)
            rand1 = random.choice('abcd')
            rand2 = random.choice('abcd'.replace(rand1,''))
            example_output = ''
            for letter in str(example):
                if letter == rand1:
                    example_output += rand2
                elif letter == rand2:
                    example_output += rand1
                else:
                    example_output += letter
            example = example_output
        output = ''
        if type_inner == 'pos':
            output = '1'
        elif type_inner == 'neg':
            output = '0'
        examples_file.write(example + " " + output + "\n")

    examples_file.close()


def merge_train_data():
    train_file = open(os.path.join(data_dir, TRAIN_EXAMPLES_NAME), 'w')

    with open(os.path.join(data_dir, POS_EXAMPLES_NAME)) as f:
        content = f.readlines()
        train_file.write(''.join(content))
    with open(os.path.join(data_dir, NEG_EXAMPLES_NAME)) as f:
        content = f.readlines()
        train_file.write(''.join(content))
    train_file.close()


if __name__ == "__main__":
    main()
